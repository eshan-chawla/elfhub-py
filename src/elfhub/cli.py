"""The ``elfhub`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from elfhub import __version__
from elfhub.agent import Elf
from elfhub.config import config_schema
from elfhub.elfdir import CONFIG_FILE, scaffold
from elfhub.errors import ElfhubError
from elfhub.policy.approvals import ApprovalBroker, ApprovalRequest, ApproveAll, CallbackBroker

_EXIT_CODES = {"completed": 0, "suspended": 3}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ElfhubError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elfhub", description="Build and run elves.")
    parser.add_argument("--version", action="version", version=f"elfhub {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="create a new elf folder")
    new.add_argument("path", help="where to create the elf")
    new.add_argument("--name", help="the elf's name (default: the folder name)")
    new.set_defaults(handler=_new)

    run = commands.add_parser("run", help="give an elf a task")
    run.add_argument("path", help="the elf's folder")
    run.add_argument("task", nargs="+", help="what to ask the elf")
    run.add_argument("--session", help="continue an earlier session")
    run.add_argument("--yes", action="store_true", help="approve every tool call without asking")
    run.add_argument("--json", action="store_true", help="print the full run result as JSON")
    run.set_defaults(handler=_run)

    schema = commands.add_parser("schema", help=f"print the JSON Schema of {CONFIG_FILE}")
    schema.set_defaults(handler=_schema)
    return parser


def _new(args: argparse.Namespace) -> int:
    name = args.name or Path(args.path).resolve().name
    root = scaffold(args.path, name=name)
    print(f"Created elf {name!r} in {root}")
    print(f'Next: set `llm` in {root / CONFIG_FILE}, then run: elfhub run {args.path} "hello"')
    return 0


def _run(args: argparse.Namespace) -> int:
    broker: ApprovalBroker | None = None
    if args.yes:
        broker = ApproveAll()
    elif sys.stdin.isatty():
        broker = CallbackBroker(_ask_on_console)
    elf = Elf.from_dir(args.path, broker=broker)
    result = elf.run(" ".join(args.task), session=args.session)

    if args.json:
        print(result.model_dump_json(indent=2))
    elif result.status == "completed":
        print(result.output or "")
    elif result.status == "suspended":
        names = ", ".join(p.call.name for p in result.pending_approvals)
        print(f"waiting for approval of: {names} (session {result.session_id})", file=sys.stderr)
    else:
        print(f"{result.status}: {result.error}", file=sys.stderr)
    return _EXIT_CODES.get(result.status, 1)


def _schema(args: argparse.Namespace) -> int:
    print(json.dumps(config_schema(), indent=2))
    return 0


async def _ask_on_console(request: ApprovalRequest) -> bool:
    arguments = json.dumps(request.call.arguments)
    answer = await asyncio.to_thread(input, f"Allow {request.call.name}({arguments})? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


if __name__ == "__main__":
    raise SystemExit(main())
