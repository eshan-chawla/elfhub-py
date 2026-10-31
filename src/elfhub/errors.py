"""Exception hierarchy for ElfHub."""


class ElfhubError(Exception):
    """Base class for every error raised by ElfHub."""


class PluginError(ElfhubError):
    """A plugin could not be registered, discovered or loaded."""


class DuplicatePluginError(PluginError):
    """Two plugins of the same kind claim the same name."""


class PluginNotFoundError(PluginError, KeyError):
    """No plugin of the requested kind is registered under that name."""

    def __str__(self) -> str:
        # KeyError.__str__ repr()-quotes the message; show it plainly.
        return str(self.args[0]) if self.args else ""


class ConfigError(ElfhubError):
    """An elf's configuration is invalid or inconsistent."""


class ConfigWarning(UserWarning):
    """A configuration problem that ElfHub can safely ignore, such as an unknown key."""


class InvalidSessionId(ElfhubError, ValueError):
    """A session id that is not a plain name (it could otherwise escape the store)."""


class ModelError(ElfhubError):
    """An LLM call failed. Adapters map vendor errors onto the subclasses below."""


class RateLimited(ModelError):
    """The provider asked us to slow down. Retried after ``retry_after`` seconds."""

    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientModelError(ModelError):
    """A temporary failure (overload, network, 5xx). Safe to retry."""


class ContextOverflow(ModelError):
    """The request does not fit in the model's context window."""


class AuthFailed(ModelError):
    """The provider rejected the credentials."""


class Refused(ModelError):
    """The model refused to answer."""
