"""Bounded subprocess execution with portable best-effort resource sampling."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - depends on the local runtime
    psutil = None


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    wall_seconds: float
    cpu_seconds: float | None
    peak_memory_bytes: int | None
    resource_unavailable_reason: str | None


def _sample_process_tree(process) -> tuple[float, int]:
    processes = [process, *process.children(recursive=True)]
    cpu_seconds = 0.0
    memory_bytes = 0
    for item in processes:
        try:
            times = item.cpu_times()
            cpu_seconds += float(times.user + times.system)
            memory_bytes += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return cpu_seconds, memory_bytes


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    if timeout_seconds <= 0:
        raise ValueError("command timeout must be positive")
    command = tuple(str(value) for value in argv)
    started = time.monotonic()
    child = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )

    observed_cpu: float | None = None
    peak_memory: int | None = None
    resource_reason: str | None = None
    observed = psutil.Process(child.pid) if psutil is not None else None
    if observed is None:
        resource_reason = "psutil is unavailable"

    stdout = ""
    stderr = ""
    timed_out = False
    while True:
        if observed is not None:
            try:
                cpu, memory = _sample_process_tree(observed)
                observed_cpu = max(observed_cpu or 0.0, cpu)
                peak_memory = max(peak_memory or 0, memory)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                if observed_cpu is None and peak_memory is None:
                    resource_reason = f"resource sampling unavailable: {error}"

        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            child.kill()
            stdout, stderr = child.communicate()
            break
        try:
            stdout, stderr = child.communicate(timeout=min(0.05, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

    wall_seconds = time.monotonic() - started
    return CommandResult(
        argv=command,
        exit_code=None if timed_out else child.returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        wall_seconds=wall_seconds,
        cpu_seconds=observed_cpu,
        peak_memory_bytes=peak_memory,
        resource_unavailable_reason=resource_reason,
    )
