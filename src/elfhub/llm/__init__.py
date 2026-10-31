"""The provider-neutral LLM contract."""

from elfhub.llm.protocol import (
    LLM,
    BaseLLM,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    PartDone,
    ReasoningConfig,
    ReasoningDelta,
    Stop,
    StopReason,
    TextDelta,
    UsageReport,
)
from elfhub.llm.resolve import parse_ref, resolve_llm

__all__ = [
    "LLM",
    "BaseLLM",
    "ModelCapabilities",
    "ModelEvent",
    "ModelRequest",
    "ModelResponse",
    "PartDone",
    "ReasoningConfig",
    "ReasoningDelta",
    "Stop",
    "StopReason",
    "TextDelta",
    "UsageReport",
    "parse_ref",
    "resolve_llm",
]
