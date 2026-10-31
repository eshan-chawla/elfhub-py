"""The :class:`Elf`: ElfHub's agent class."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from elfhub.errors import ConfigError, PluginError
from elfhub.events import Event, EventBus, Interceptor, Observer
from elfhub.llm.protocol import LLM
from elfhub.llm.resolve import resolve_llm
from elfhub.loop.budget import Budget
from elfhub.loop.context import RunContext
from elfhub.loop.runner import Runner, RunResult
from elfhub.persist.store import JsonlSessionStore, MemorySessionStore, SessionStore
from elfhub.plugins.module import (
    Assembly,
    ElfContext,
    Module,
    check_api_version,
    module_name,
    order_modules,
)
from elfhub.policy.approvals import ApprovalBroker, Decision, Policy
from elfhub.tools.base import Tool
from elfhub.tools.decorator import as_tool
from elfhub.types import Message, new_id

if TYPE_CHECKING:
    from elfhub.config import ElfConfig

logger = logging.getLogger("elfhub")


def _static(text: str) -> Callable[[RunContext], str]:
    return lambda _ctx: text


def _policy_layers(policy: Policy | Sequence[Policy] | None) -> list[Policy]:
    if policy is None:
        return []
    if isinstance(policy, Policy):
        return [policy]
    return list(policy)


class Elf:
    """An agent with a declared model, an identity, tools and modules.

    Configure an elf with constructor arguments, or declaratively by subclassing and
    setting class attributes (constructor arguments win)::

        class Ada(Elf):
            name = "ada"
            llm = "anthropic/claude-sonnet-5-5"
            soul = "You are Ada, a careful engineer."
            tools = (search_docs,)

    Use ``elf.run(...)`` from sync code and ``await elf.arun(...)`` from async code.
    Each call starts the elf's modules and stops them afterwards; wrap several calls
    in ``async with elf:`` to keep them running.
    """

    name: str = "elf"
    llm: str | LLM | None = None
    soul: str | None = None
    instructions: str | None = None
    tools: Sequence[Tool | Callable[..., Any]] = ()
    modules: Sequence[Module | type[Module] | str] = ()
    policy: Policy | Sequence[Policy] | None = None
    broker: ApprovalBroker | None = None
    budget: Budget | None = None

    def __init__(
        self,
        *,
        name: str | None = None,
        llm: str | LLM | None = None,
        llm_options: Mapping[str, Any] | None = None,
        soul: str | None = None,
        instructions: str | None = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        modules: Sequence[Module | type[Module] | str] | None = None,
        module_config: Mapping[str, Mapping[str, Any]] | None = None,
        policy: Policy | Sequence[Policy] | None = None,
        broker: ApprovalBroker | None = None,
        budget: Budget | None = None,
        home: str | Path | None = None,
        store: SessionStore | None = None,
        config: ElfConfig | None = None,
    ) -> None:
        cls = type(self)
        self.name = name if name is not None else cls.name
        self.llm = llm if llm is not None else cls.llm
        self.llm_options = dict(llm_options or {})
        self.soul = soul if soul is not None else cls.soul
        self.instructions = instructions if instructions is not None else cls.instructions
        self.tools = list(tools if tools is not None else cls.tools)
        self.modules = list(modules if modules is not None else cls.modules)
        self.module_config = {key: dict(value) for key, value in (module_config or {}).items()}
        self.policy = policy if policy is not None else cls.policy
        self.broker = broker if broker is not None else cls.broker
        self.budget = budget if budget is not None else cls.budget
        self.home = Path(home) if home is not None else None
        if store is not None:
            self.store: SessionStore = store
        elif self.home is not None:
            self.store = JsonlSessionStore(self.home / ".elf" / "sessions")
        else:
            self.store = MemorySessionStore()
        self.config = config
        """The parsed elf.toml, when the elf was loaded from a folder."""
        self.bus = EventBus()
        self._runner: Runner | None = None
        self._modules: list[Module[Any]] = []
        self._contexts: list[ElfContext] = []
        self._active_runs = 0
        self._auto_started = False
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def from_dir(cls, path: str | Path, **overrides: Any) -> Self:
        """Load an elf from its folder (elf.toml, SOUL.md, AGENTS.md, tools/).

        Keyword arguments override what the folder declares. Loading imports the
        folder's ``tools/*.py`` files, so only load elves you trust.
        """
        from elfhub.elfdir import load_elf_kwargs

        kwargs = load_elf_kwargs(Path(path))
        # The folder adds to what the class declares; it never silently drops it.
        if "tools" in kwargs:
            kwargs["tools"] = [*cls.tools, *kwargs["tools"]]
        if "modules" in kwargs:
            kwargs["modules"] = [*cls.modules, *kwargs["modules"]]
        if "policy" in kwargs:
            kwargs["policy"] = [*_policy_layers(cls.policy), kwargs["policy"]]
        return cls(**{**kwargs, **overrides})

    # -- running ---------------------------------------------------------------

    async def arun(
        self,
        input: str | Message | Sequence[Message],
        *,
        session: str | None = None,
        budget: Budget | None = None,
    ) -> RunResult:
        """Run the elf on ``input``, in a new session unless ``session`` is given."""
        async with self._running() as runner:
            return await runner.run(input, session_id=session, budget=budget or self.budget)

    async def aresume(
        self,
        session: str,
        decisions: Mapping[str, Decision | bool],
        *,
        budget: Budget | None = None,
    ) -> RunResult:
        """Continue a suspended session with approval decisions keyed by tool call id."""
        async with self._running() as runner:
            return await runner.resume(session, decisions, budget=budget or self.budget)

    async def astream(
        self,
        input: str | Message | Sequence[Message],
        *,
        session: str | None = None,
        budget: Budget | None = None,
    ) -> AsyncIterator[Event]:
        """Run the elf, yielding each event of the run as it happens."""
        run_id = new_id("run")
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        def observe(event: Event) -> None:
            if event.run_id == run_id:
                queue.put_nowait(event)

        unsubscribe = self.bus.on("*", observe)
        try:
            async with self._running() as runner:
                task = asyncio.create_task(
                    runner.run(
                        input, session_id=session, run_id=run_id, budget=budget or self.budget
                    )
                )
                task.add_done_callback(lambda _task: queue.put_nowait(None))
                try:
                    while (event := await queue.get()) is not None:
                        yield event
                    task.result()
                finally:
                    if not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
        finally:
            unsubscribe()

    def run(
        self,
        input: str | Message | Sequence[Message],
        *,
        session: str | None = None,
        budget: Budget | None = None,
    ) -> RunResult:
        """Synchronous :meth:`arun`. Inside a running event loop it uses a worker thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(input, session=session, budget=budget))
        if self._runner is not None:
            raise RuntimeError(
                "this elf is running inside an event loop; use `await elf.arun(...)`"
            )
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, self.arun(input, session=session, budget=budget)
            ).result()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """What calling the elf does. Runs it by default; override to customise."""
        return self.run(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)

    # -- hooks -----------------------------------------------------------------

    def on(self, event_type: str, fn: Observer, *, priority: int = 0) -> Callable[[], None]:
        """Observe events (``"*"`` for all). Returns an unsubscribe function."""
        return self.bus.on(event_type, fn, priority=priority)

    def intercept(self, hook: str, fn: Interceptor, *, priority: int = 0) -> Callable[[], None]:
        """Intercept a hook such as ``before_tool_call``. Returns an unsubscribe function."""
        return self.bus.intercept(hook, fn, priority=priority)

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        """Resolve the model, set up and start modules, and keep them running until
        :meth:`stop`. Runs start the elf automatically when needed."""
        async with self._lifecycle_lock():
            await self._start_locked()
            self._auto_started = False

    async def stop(self) -> None:
        """Stop modules in reverse order and remove their hooks."""
        async with self._lifecycle_lock():
            await self._stop_locked()

    async def __aenter__(self) -> Elf:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    @asynccontextmanager
    async def _running(self) -> AsyncGenerator[Runner, None]:
        """Use the running elf, starting it first if needed. An elf started here is
        stopped when its last concurrent run finishes."""
        lock = self._lifecycle_lock()
        async with lock:
            if self._runner is None:
                await self._start_locked()
                self._auto_started = True
            self._active_runs += 1
            runner = self._runner
        try:
            assert runner is not None
            yield runner
        finally:
            async with lock:
                self._active_runs -= 1
                if self._active_runs == 0 and self._auto_started:
                    await self._stop_locked()

    def _lifecycle_lock(self) -> asyncio.Lock:
        # asyncio locks belong to one event loop; sync run() uses a fresh loop each time.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def _start_locked(self) -> None:
        if self._runner is not None:
            return
        if self.llm is None:
            raise ConfigError(
                f'elf {self.name!r} has no llm declared: set llm = "provider/model" in elf.toml, '
                "or pass llm=... to Elf()"
            )
        llm = resolve_llm(self.llm, **self.llm_options)
        modules = order_modules([self._make_module(entry) for entry in self.modules])
        assembly = Assembly(tools=[as_tool(t) for t in self.tools], policies=self._policies())
        for text in (self.soul, self.instructions):
            if text:
                assembly.context.append(_static(text))

        contexts: list[ElfContext] = []
        started: list[Module[Any]] = []
        try:
            for module in modules:
                ctx = ElfContext(
                    name=self.name,
                    home=self.home,
                    store=self.store,
                    bus=self.bus,
                    llms={"main": llm},
                    assembly=assembly,
                )
                contexts.append(ctx)
                module.setup(ctx)
            tools = self._index_tools(assembly.tools)
            for module in modules:
                await module.start()
                started.append(module)
        except BaseException:
            await self._stop_modules(started)
            for ctx in contexts:
                ctx.dispose()
            raise

        self._runner = Runner(
            llm=llm,
            tools=tools,
            store=self.store,
            bus=self.bus,
            policies=assembly.policies,
            broker=self.broker,
            context=assembly.context,
        )
        self._modules = modules
        self._contexts = contexts

    async def _stop_locked(self) -> None:
        if self._runner is None:
            return
        await self._stop_modules(self._modules)
        for ctx in self._contexts:
            ctx.dispose()
        self._runner, self._modules, self._contexts = None, [], []

    # -- helpers -----------------------------------------------------------------

    def _make_module(self, entry: Module[Any] | type[Module[Any]] | str) -> Module[Any]:
        if isinstance(entry, Module):
            check_api_version(type(entry))
            return entry
        if isinstance(entry, str):
            from elfhub.plugins import modules as registry

            # Third-party factories are untrusted: check what they actually return.
            factory = cast(Callable[..., object], registry.get(entry))
            if isinstance(factory, type):
                check_api_version(factory)
            created = factory(self.module_config.get(entry, {}))
            if not isinstance(created, Module):
                raise PluginError(f"module plugin {entry!r} did not produce a Module")
            module = cast(Module[Any], created)
            check_api_version(type(module))
            return module
        check_api_version(entry)
        return entry(self.module_config.get(getattr(entry, "name", ""), {}))

    def _policies(self) -> list[Policy]:
        return _policy_layers(self.policy)

    @staticmethod
    def _index_tools(tools: Sequence[Tool]) -> dict[str, Tool]:
        indexed: dict[str, Tool] = {}
        for tool in tools:
            if tool.spec.name in indexed:
                raise ConfigError(f"two tools are both named {tool.spec.name!r}")
            indexed[tool.spec.name] = tool
        return indexed

    @staticmethod
    async def _stop_modules(modules: Sequence[Module]) -> None:
        for module in reversed(modules):
            try:
                await module.stop()
            except Exception:
                logger.exception("module %r failed to stop", module_name(module))
