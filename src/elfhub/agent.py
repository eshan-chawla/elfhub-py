"""Base agent class for ElfHub."""


class Elf:
    """Base agent class. All agents in ElfHub derive from this."""

    def invoke(self, *args, **kwargs):
        """Respond to an invocation."""
        return "Hi !"

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
