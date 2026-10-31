"""Test helpers for elves and plugins: a scripted LLM and conformance checks."""

from elfhub.testing.conformance import (
    check_auth_error,
    check_context_overflow,
    check_llm,
    check_parallel_tool_calls,
    check_reasoning_round_trip,
    check_stream,
    check_text_reply,
    check_tool_round_trip,
    check_usage_reported,
)
from elfhub.testing.llm import ScriptedLLM, Step, call

__all__ = [
    "ScriptedLLM",
    "Step",
    "call",
    "check_auth_error",
    "check_context_overflow",
    "check_llm",
    "check_parallel_tool_calls",
    "check_reasoning_round_trip",
    "check_stream",
    "check_text_reply",
    "check_tool_round_trip",
    "check_usage_reported",
]
