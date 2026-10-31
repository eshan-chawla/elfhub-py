"""Provider-neutral message types shared by every part of ElfHub.

LLM adapters translate between these types and a vendor's wire format, so nothing
outside an adapter ever sees vendor-specific structures.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Trust = Literal["owner", "system", "self", "untrusted"]
"""Who a piece of content came from: the elf's owner, the framework, the elf itself,
or an outside source (web pages, strangers, MCP servers...)."""

Role = Literal["user", "assistant", "tool"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class _Model(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")


class TextPart(_Model):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Model):
    """An image, either inline (``data``) or by reference (``url``)."""

    type: Literal["image"] = "image"
    media_type: str
    data: bytes | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.data is None) == (self.url is None):
            raise ValueError("an image needs exactly one of `data` or `url`")
        return self


class ToolCallPart(_Model):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict[str, Any])


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ToolResultPart(_Model):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: list[ContentPart]
    is_error: bool = False


class ReasoningPart(_Model):
    """Model reasoning. ``opaque`` holds provider data (e.g. signatures) that must be
    sent back unchanged to the same provider and dropped for any other."""

    type: Literal["reasoning"] = "reasoning"
    text: str | None = None
    provider: str
    opaque: dict[str, Any] = Field(default_factory=dict[str, Any])


Part = Annotated[
    TextPart | ImagePart | ToolCallPart | ToolResultPart | ReasoningPart,
    Field(discriminator="type"),
]


class Message(_Model):
    id: str = Field(default_factory=lambda: new_id("msg"))
    role: Role
    parts: list[Part]
    trust: Trust = "owner"

    @classmethod
    def user(cls, text: str, *, trust: Trust = "owner") -> Message:
        return cls(role="user", parts=[TextPart(text=text)], trust=trust)

    @property
    def text(self) -> str:
        return "\n".join(part.text for part in self.parts if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    @property
    def tool_results(self) -> list[ToolResultPart]:
        return [part for part in self.parts if isinstance(part, ToolResultPart)]


class Usage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )
