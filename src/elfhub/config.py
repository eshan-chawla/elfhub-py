"""The ``elf.toml`` manifest: an elf's declarative configuration.

Unknown keys are handled by one rule. In the safety sections (``[budget]`` and
``[policy]``) they are errors, because ignoring a misspelt limit or rule would
silently loosen it. Anywhere else they are reported as a :class:`ConfigWarning`
(with a "did you mean" hint) and ignored, so an elf written for a newer ElfHub still
loads. Settings owned by other programs belong in ``[extensions.<owner>]``, which is
kept as-is. ElfHub never rewrites elf.toml itself, so nothing in the file is lost.
"""

from __future__ import annotations

import difflib
import tomllib
import warnings
from pathlib import Path
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from elfhub.errors import ConfigError, ConfigWarning
from elfhub.loop.budget import Budget
from elfhub.policy.approvals import Policy, Rule

FORMAT_VERSION = 1
"""The newest ``format_version`` of elf.toml this version of ElfHub can read."""


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main: str | None = None
    """The model the elf thinks with, as ``"provider/model"``."""
    options: dict[str, Any] = Field(default_factory=dict[str, Any])
    """Passed to the provider's adapter."""


def _hint(key: str, known: Any) -> str:
    match = difflib.get_close_matches(key, list(known), n=1)
    return f" (did you mean {match[0]!r}?)" if match else ""


def _unknown(table: dict[str, Any], model: type[BaseModel]) -> list[str]:
    return [key for key in table if key not in model.model_fields]


def _warn_unknown(key: str, where: str, model: type[BaseModel]) -> None:
    warnings.warn(
        f"unknown setting {key!r}{where}{_hint(key, model.model_fields)} is ignored",
        ConfigWarning,
        stacklevel=4,
    )


def _refuse_unknown(table: dict[str, Any], where: str, model: type[BaseModel]) -> None:
    # Safety settings: ignoring a misspelt key would silently loosen a limit or rule.
    for key in _unknown(table, model):
        raise ValueError(f"unknown setting {key!r} in {where}{_hint(key, model.model_fields)}")


class ElfConfig(BaseModel):
    format_version: int = FORMAT_VERSION
    name: str
    version: str = "0.1.0"
    description: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: list[str] = Field(default_factory=list[str])
    """Tools to enable from installed plugins, by name (``tools/*.py`` load automatically)."""
    modules: dict[str, dict[str, Any]] = Field(default_factory=dict[str, dict[str, Any]])
    """Modules to enable, by name, each with its config table."""
    policy: Policy = Field(default_factory=Policy)
    budget: Budget = Field(default_factory=Budget)
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict[str, dict[str, Any]])
    """Settings owned by other programs (e.g. a hosted runtime), namespaced by owner."""

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(cast(dict[str, Any], raw))
        for key in _unknown(data, cls):
            _warn_unknown(key, "", cls)
            del data[key]
        if isinstance(data.get("llm"), dict):
            llm = data["llm"] = dict(cast(dict[str, Any], data["llm"]))
            for key in _unknown(llm, LLMConfig):
                _warn_unknown(key, " in [llm]", LLMConfig)
                del llm[key]
        if isinstance(data.get("budget"), dict):
            _refuse_unknown(cast(dict[str, Any], data["budget"]), "[budget]", Budget)
        if isinstance(data.get("policy"), dict):
            policy = cast(dict[str, Any], data["policy"])
            _refuse_unknown(policy, "[policy]", Policy)
            rules = policy.get("rules")
            for number, rule in enumerate(
                cast(list[Any], rules) if isinstance(rules, list) else []
            ):
                if isinstance(rule, dict):
                    _refuse_unknown(cast(dict[str, Any], rule), f"policy rule {number + 1}", Rule)
        return data

    @field_validator("llm", mode="before")
    @classmethod
    def _model_string_shorthand(cls, value: Any) -> Any:
        return {"main": value} if isinstance(value, str) else value

    @field_validator("format_version")
    @classmethod
    def _supported_format(cls, value: int) -> int:
        if value > FORMAT_VERSION:
            raise ValueError(
                f"this elf was made by a newer elfhub (format {value}; this version reads up to "
                f"{FORMAT_VERSION}); upgrade elfhub to use it"
            )
        return value


def load_config(path: str | Path) -> ElfConfig:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigWarning)
        try:
            config = ElfConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    for warning in caught:
        if issubclass(warning.category, ConfigWarning):
            warnings.warn(f"{path}: {warning.message}", ConfigWarning, stacklevel=2)
        else:
            warnings.warn(warning.message, warning.category, stacklevel=2)
    return config


def config_schema() -> dict[str, Any]:
    """JSON Schema for elf.toml, e.g. for editors and the hosted runtime's UI."""
    return ElfConfig.model_json_schema()
