"""ElfHub: an open-source framework for building human-like agents (elves)."""

__version__ = "0.0.2"

from elfhub.agent import Elf
from elfhub.loop.budget import Budget
from elfhub.loop.runner import RunResult
from elfhub.plugins.module import ElfContext, Module
from elfhub.policy.approvals import Decision, Policy, Rule
from elfhub.tools import tool
from elfhub.types import Message

__all__ = [
    "Budget",
    "Decision",
    "Elf",
    "ElfContext",
    "Message",
    "Module",
    "Policy",
    "Rule",
    "RunResult",
    "__version__",
    "tool",
]
