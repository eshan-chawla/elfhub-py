from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError

from elfhub.errors import ConfigError, ModelError, PluginNotFoundError, RateLimited
from elfhub.llm import (
    BaseLLM,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    PartDone,
    Stop,
    TextDelta,
    UsageReport,
    resolve_llm,
)
from elfhub.plugins import Registry
from elfhub.testing import ScriptedLLM, call
from elfhub.types import Message, TextPart, ToolCallPart, Usage


class FixedStreamLLM(BaseLLM):
    ref = "fixed/stream"

    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        for event in self.events:
            yield event


def request() -> ModelRequest:
    return ModelRequest(messages=[Message.user("hi")])


async def test_generate_folds_completed_parts_usage_and_stop_reason():
    llm = FixedStreamLLM(
        [
            TextDelta(text="Let me "),
            TextDelta(text="check."),
            PartDone(part=TextPart(text="Let me check.")),
            UsageReport(usage=Usage(input_tokens=7)),
            PartDone(part=ToolCallPart(id="c1", name="lookup", arguments={"q": "x"})),
            UsageReport(usage=Usage(output_tokens=3)),
            Stop(reason="tool_use"),
        ]
    )

    response = await llm.generate(request())

    assert response.message.role == "assistant"
    assert response.message.parts == [
        TextPart(text="Let me check."),
        ToolCallPart(id="c1", name="lookup", arguments={"q": "x"}),
    ]
    assert response.usage == Usage(input_tokens=7, output_tokens=3)
    assert response.stop_reason == "tool_use"


async def test_model_output_is_marked_as_coming_from_the_elf_itself():
    llm = FixedStreamLLM([PartDone(part=TextPart(text="ok")), Stop(reason="end")])

    response = await llm.generate(request())

    assert response.message.trust == "self"


async def test_a_stream_without_stop_is_a_model_error():
    llm = FixedStreamLLM([PartDone(part=TextPart(text="cut off"))])

    with pytest.raises(ModelError, match="stop"):
        await llm.generate(request())


async def test_base_llm_cannot_count_tokens_by_default():
    assert await FixedStreamLLM([]).count_tokens(request()) is None


class EchoModel(BaseLLM):
    def __init__(self, model: str, **options: Any) -> None:
        self.ref = f"echo/{model}"
        self.model = model
        self.options = options

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield Stop(reason="end")


def echo_registry() -> Registry[BaseLLM]:
    reg: Registry[BaseLLM] = Registry("llm")
    reg.register(EchoModel, name="echo")
    return reg


def test_resolve_passes_llm_objects_through():
    llm = EchoModel("x")

    assert resolve_llm(llm) is llm


def test_resolve_builds_the_provider_adapter_with_the_model_id_and_options():
    llm = resolve_llm("echo/big-model", registry=echo_registry(), temperature=0.1)

    assert isinstance(llm, EchoModel)
    assert (llm.model, llm.options) == ("big-model", {"temperature": 0.1})


def test_resolve_keeps_slashes_inside_the_model_id():
    llm = resolve_llm("echo/vendor/model-7", registry=echo_registry())

    assert isinstance(llm, EchoModel)
    assert llm.model == "vendor/model-7"


def test_resolve_unknown_provider_suggests_installing_an_adapter():
    with pytest.raises(PluginNotFoundError) as info:
        resolve_llm("acme/model", registry=echo_registry())

    message = str(info.value)
    assert "acme" in message and "echo" in message and "pip install" in message


@pytest.mark.parametrize("ref", ["no-slash", "/model", "provider/"])
def test_resolve_rejects_malformed_references(ref: str):
    with pytest.raises(ConfigError, match="provider/model"):
        resolve_llm(ref, registry=echo_registry())


async def test_scripted_llm_plays_its_steps_in_order_and_records_requests():
    llm = ScriptedLLM(call("lookup", q="x"), "done")

    first = await llm.generate(request())
    second = await llm.generate(ModelRequest(messages=[Message.user("again")]))

    assert first.stop_reason == "tool_use"
    assert first.message.tool_calls[0].name == "lookup"
    assert first.message.tool_calls[0].arguments == {"q": "x"}
    assert (second.message.text, second.stop_reason) == ("done", "end")
    assert [r.messages[0].text for r in llm.requests] == ["hi", "again"]


async def test_scripted_llm_raises_exception_steps():
    llm = ScriptedLLM(RateLimited("slow down", retry_after=1.0))

    with pytest.raises(RateLimited):
        await llm.generate(request())


async def test_scripted_llm_steps_can_depend_on_the_request():
    llm = ScriptedLLM(lambda req: f"you said {req.messages[-1].text}")

    response = await llm.generate(request())

    assert response.message.text == "you said hi"


async def test_scripted_llm_fails_loudly_when_it_runs_out_of_steps():
    llm = ScriptedLLM("only one")
    await llm.generate(request())

    with pytest.raises(AssertionError, match="no step"):
        await llm.generate(request())


def test_call_helper_generates_distinct_ids():
    assert call("a").id != call("a").id


def test_capabilities_cannot_be_changed_in_place():
    capabilities = ModelCapabilities()

    with pytest.raises(ValidationError):
        capabilities.vision = True
