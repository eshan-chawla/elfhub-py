"""Payloads passed to interceptors.

An interceptor registered with ``on``/``intercept`` receives one of these and
returns ``None`` (no change), a modified copy (``dataclasses.replace(payload, ...)``),
or a :class:`~elfhub.events.Veto`.

=====================  ===================  =========================================
Hook                   Payload              A veto...
=====================  ===================  =========================================
``before_model_call``  :class:`ModelCall`        fails the run
``before_tool_call``   :class:`ToolInvocation`   refuses the call (the model is told)
``after_tool_call``    :class:`ToolOutcome`      withholds the result (the model is told)
=====================  ===================  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass

from elfhub.llm.protocol import ModelRequest
from elfhub.tools.base import ToolResult, ToolSpec
from elfhub.types import ToolCallPart

BEFORE_MODEL_CALL = "before_model_call"
BEFORE_TOOL_CALL = "before_tool_call"
AFTER_TOOL_CALL = "after_tool_call"


@dataclass(frozen=True)
class ModelCall:
    request: ModelRequest
    session_id: str
    run_id: str


@dataclass(frozen=True)
class ToolInvocation:
    """A tool call about to run. Replace ``call`` to rewrite its arguments."""

    call: ToolCallPart
    spec: ToolSpec
    session_id: str
    run_id: str


@dataclass(frozen=True)
class ToolOutcome:
    """A finished tool call. Replace ``result`` to change what the model sees."""

    call: ToolCallPart
    spec: ToolSpec
    result: ToolResult
    session_id: str
    run_id: str
