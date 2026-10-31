import pytest
from pydantic import ValidationError

from elfhub.types import (
    ImagePart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)


def test_every_part_type_round_trips_through_json():
    message = Message(
        role="assistant",
        parts=[
            TextPart(text="hello"),
            ImagePart(media_type="image/png", data=b"\x89PNG\r\n\x1a\n\xff\x00"),
            ImagePart(media_type="image/jpeg", url="https://example.com/cat.jpg"),
            ToolCallPart(id="call_1", name="add", arguments={"a": 1, "b": [2, 3]}),
            ToolResultPart(call_id="call_1", content=[TextPart(text="5")], is_error=False),
            ReasoningPart(text="thinking", provider="acme", opaque={"signature": "abc"}),
        ],
        trust="untrusted",
    )

    restored = Message.model_validate_json(message.model_dump_json())

    assert restored == message
    assert isinstance(restored.parts[1], ImagePart)
    assert restored.parts[1].data == b"\x89PNG\r\n\x1a\n\xff\x00"


def test_image_needs_exactly_one_of_data_or_url():
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/png")
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/png", data=b"x", url="https://example.com/x.png")


def test_message_text_joins_only_text_parts():
    message = Message(
        role="assistant",
        parts=[
            TextPart(text="first"),
            ToolCallPart(id="c1", name="noop"),
            TextPart(text="second"),
        ],
    )

    assert message.text == "first\nsecond"


def test_message_exposes_its_tool_calls_in_order():
    message = Message(
        role="assistant",
        parts=[
            ToolCallPart(id="c1", name="a"),
            TextPart(text="between"),
            ToolCallPart(id="c2", name="b"),
        ],
    )

    assert [call.id for call in message.tool_calls] == ["c1", "c2"]


def test_user_message_shortcut():
    message = Message.user("hi there")

    assert message.role == "user"
    assert message.trust == "owner"
    assert message.parts == [TextPart(text="hi there")]


def test_messages_get_unique_ids():
    assert Message.user("a").id != Message.user("a").id


def test_usage_adds_up():
    total = Usage(input_tokens=10, output_tokens=5) + Usage(
        input_tokens=1, output_tokens=2, cost_usd=0.25
    )

    assert total == Usage(input_tokens=11, output_tokens=7, cost_usd=0.25)


def test_usage_cost_stays_unknown_when_no_side_knows_it():
    assert (Usage(input_tokens=1) + Usage(output_tokens=1)).cost_usd is None
