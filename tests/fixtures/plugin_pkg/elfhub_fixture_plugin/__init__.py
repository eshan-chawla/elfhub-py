"""A minimal third-party plugin: one LLM adapter, one module and one tool."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from pydantic import BaseModel

from elfhub.llm import BaseLLM, ModelEvent, ModelRequest, PartDone, Stop
from elfhub.plugins.module import ElfContext, Module
from elfhub.tools import tool
from elfhub.types import TextPart, ToolCallPart, new_id


class FixtureLLM(BaseLLM):
    """Calls fixture_shout once on the user's text, then reports what it got back."""

    def __init__(self, model: str, **options: Any) -> None:
        self.ref = f"fixture/{model}"
        self.model = model

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        last = request.messages[-1]
        if last.role == "tool":
            shouted = last.tool_results[0].content[0]
            text = shouted.text if isinstance(shouted, TextPart) else ""
            system = " ".join(part.text for part in request.system)
            yield PartDone(part=TextPart(text=f"{self.model} | {system} | {text}"))
            yield Stop(reason="end")
            return
        # Tool call ids must be unique across turns, not just within one reply.
        call = ToolCallPart(id=new_id("call"), name="fixture_shout", arguments={"text": last.text})
        yield PartDone(part=call)
        yield Stop(reason="tool_use")


class GreeterConfig(BaseModel):
    greeting: str = "Hi"


class FixtureGreeter(Module[GreeterConfig]):
    name: ClassVar[str] = "fixture-greeter"

    def setup(self, ctx: ElfContext) -> None:
        greeting = self.config.greeting
        ctx.add_context(lambda _ctx: f"{greeting} from the fixture plugin.")


@tool(name="fixture_shout")
def shout(text: str) -> str:
    """Shout the text back."""
    return text.upper()
