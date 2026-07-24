#!/usr/bin/env python3
"""Run one U2 trainer in its own process group and forward stop requests safely."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHILD_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _terminate_unpublished_child(child: subprocess.Popen[Any]) -> None:
    """Reap a child whose ownership state could not be published."""

    if child.poll() is not None:
        child.wait()
        return
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=CHILD_SHUTDOWN_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=CHILD_SHUTDOWN_TIMEOUT_SECONDS)


def run_supervised(command: list[str], *, state_path: Path) -> int:
    """Return the child status after translating INT/TERM into trainer SIGINT."""

    if not command:
        raise ValueError("supervisor requires a trainer command")
    state_file = state_path.expanduser()
    if not state_file.is_absolute():
        raise ValueError("supervisor state path must be absolute")
    parent = state_file.parent.resolve(strict=True)
    state_file = parent / state_file.name
    if state_file.exists() or state_file.is_symlink():
        raise ValueError(f"refusing to replace supervisor state: {state_file}")
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    state = {
        "schema_version": 1,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "supervisor_pid": os.getpid(),
        "trainer_pid": child.pid,
        "state": "running",
        "forwarded_signal": None,
        "exit_status": None,
    }
    try:
        _atomic_write(state_file, state)
    except BaseException:
        # The launcher cannot safely own or signal a trainer until this
        # authenticated PID record is durable. Never leave the just-created
        # process group alive when publication fails.
        _terminate_unpublished_child(child)
        raise
    forwarded = False

    def request_stop(received: int, _frame: Any) -> None:
        nonlocal forwarded
        signal_name = signal.Signals(received).name
        state["updated_at"] = _utc_now()
        state["state"] = "stop_requested"
        state["forwarded_signal"] = signal_name
        if child.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGINT if not forwarded else signal.SIGTERM)
        with suppress(OSError):
            _atomic_write(state_file, state)
        forwarded = True

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        status = child.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    state["updated_at"] = _utc_now()
    state["state"] = "exited"
    state["exit_status"] = status
    _atomic_write(state_file, state)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("supervisor requires a command after --")
    try:
        status = run_supervised(command, state_path=args.state_path)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    raise SystemExit(status)


if __name__ == "__main__":
    main()
