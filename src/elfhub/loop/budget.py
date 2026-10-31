"""Run budgets. Limits are checked between steps, so a turn already under way finishes."""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from elfhub.types import Usage


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = 50
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_s: float | None = None


class BudgetTracker:
    def __init__(self, budget: Budget, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self.turns = 0
        self.tool_calls = 0
        self.usage = Usage()
        self._clock = clock
        self._started = clock()

    def charge_turn(self) -> None:
        self.turns += 1

    def charge_tool_calls(self, count: int) -> None:
        self.tool_calls += count

    def charge_usage(self, usage: Usage) -> None:
        self.usage += usage

    def remaining_wall_s(self) -> float | None:
        """Seconds left before ``max_wall_s``, or ``None`` without a wall-clock limit."""
        if self.budget.max_wall_s is None:
            return None
        return self.budget.max_wall_s - (self._clock() - self._started)

    def exceeded(self) -> str | None:
        """Why the budget is used up, or ``None`` if there is room left."""
        b = self.budget
        checks: list[tuple[str, float | None, float]] = [
            ("max_turns", b.max_turns, self.turns),
            ("max_tool_calls", b.max_tool_calls, self.tool_calls),
            ("max_tokens", b.max_tokens, self.usage.input_tokens + self.usage.output_tokens),
            ("max_cost_usd", b.max_cost_usd, self.usage.cost_usd or 0.0),
            ("max_wall_s", b.max_wall_s, self._clock() - self._started),
        ]
        for name, limit, used in checks:
            if limit is not None and used >= limit:
                return f"{name} ({limit}) reached"
        return None
