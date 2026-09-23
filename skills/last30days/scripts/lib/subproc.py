"""Subprocess helpers: safe timeout + process-group cleanup.

Used by bird_x.py (Node.js Bird search) and youtube_yt.py (yt-dlp search
and transcript download). Both need the same os.setsid/killpg cleanup
dance on timeout to avoid orphaning child processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence


class SubprocTimeout(Exception):
    """Raised when a subprocess exceeds its timeout and is killed."""


def _register_child(pid: int) -> None:
    """Best-effort registration of a spawned child in the engine registry.

    last30days.py keeps a process-wide set of live child PIDs whose
    process groups are SIGTERMed by its SIGTERM handler / atexit cleanup
    when the engine itself is terminated (e.g. MCP timeout). Each child
    runs in its own pgid via os.setsid, so a group kill aimed at the
    engine can never reach them — only this registry can. Registration
    lives here instead of at each call site so every present and future
    caller (bird, yt-dlp, transcribe, digg, ...) is covered. The lazy
    import mirrors bird_x: lib/ must stay importable without the engine
    entrypoint, and last30days imports lib/, so a top-level import would
    be circular. Failures never break the run.
    """
    try:
        from last30days import register_child_pid
    except ImportError:
        return
    try:
        register_child_pid(pid)
    except Exception:
        pass


def _unregister_child(pid: int) -> None:
    """Best-effort removal from the engine registry (see _register_child)."""
    try:
        from last30days import unregister_child_pid
    except ImportError:
        return
    try:
        unregister_child_pid(pid)
    except Exception:
        pass


@dataclass
class SubprocResult:
    """Result of a subprocess run that captured stdout and stderr."""

    returncode: int
    stdout: str
    stderr: str


def run_with_timeout(
    cmd: Sequence[str],
    *,
    timeout: int,
    env: Optional[dict] = None,
    on_pid: Optional[callable] = None,
) -> SubprocResult:
    """Run a subprocess with process-group cleanup on timeout.

    Spawns ``cmd`` inside its own process group via ``os.setsid`` where
    available. If ``communicate(timeout=...)`` raises ``TimeoutExpired``,
    signals ``SIGTERM`` to the entire group, falls back to ``proc.kill()``
    if the signal fails, then waits up to 5 seconds for cleanup, and
    raises ``SubprocTimeout``.

    Args:
        cmd: Command and arguments to spawn.
        timeout: Timeout in seconds passed to ``communicate()``.
        env: Optional environment dict. If None, inherits parent env.
        on_pid: Optional callable invoked with the child PID right after
            spawn (kept for backward compatibility; the child is now also
            auto-registered in the engine registry — see _register_child).
            Exceptions raised by the callback are suppressed.

    Returns:
        SubprocResult with returncode, stdout, and stderr as strings.

    Raises:
        SubprocTimeout: If the process exceeded ``timeout``.
        FileNotFoundError: If the executable is not found.
        OSError: For other spawn failures.
    """
    preexec = os.setsid if hasattr(os, "setsid") else None

    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        preexec_fn=preexec,
        env=env,
    )

    if on_pid is not None:
        try:
            on_pid(proc.pid)
        except Exception:
            pass

    _register_child(proc.pid)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError, AttributeError):
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Child ignored SIGTERM (or our killpg lost the race); escalate.
                # Guard killpg/getpgid the same way the SIGTERM path above does:
                # they are POSIX-only and raise AttributeError on Windows. The
                # primary path was hardened in #552; this mirrors that guard on the
                # escalation path (added later in #433) so the same crash can't
                # re-surface here (#588).
                try:
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, PermissionError, OSError, AttributeError):
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass  # process unkillable (e.g. D-state); leave as zombie
            raise SubprocTimeout(f"Command {cmd[0]} timed out after {timeout}s")
    finally:
        _unregister_child(proc.pid)

    return SubprocResult(
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )
