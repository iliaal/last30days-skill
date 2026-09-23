"""Subprocess helpers: safe timeout + process-group cleanup.

Used by bird_x.py (Node.js Bird search) and youtube_yt.py (yt-dlp search
and transcript download). Both need the same os.setsid/killpg cleanup
dance on timeout to avoid orphaning child processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, Sequence


class SubprocTimeout(Exception):
    """Raised when a subprocess exceeds its timeout and is killed."""


# Live run_with_timeout children, process-wide. Each child is a session
# leader (os.setsid), so a group kill aimed at the engine never reaches it;
# the engine's SIGTERM handler and atexit hook drain this set instead. It
# lives here, not in last30days.py, because the entrypoint runs as
# __main__: importing it by name from lib/ executes a second module copy
# with its own empty registry, and on a worker thread that copy cannot
# install its signal handler either. RLock because cleanup_children runs
# inside a signal handler on the main thread, which may already hold the
# lock in register_child_pid.
_child_pids: set[int] = set()
_child_pids_lock = threading.RLock()


def register_child_pid(pid: int) -> None:
    with _child_pids_lock:
        _child_pids.add(pid)


def unregister_child_pid(pid: int) -> None:
    with _child_pids_lock:
        _child_pids.discard(pid)


def cleanup_children() -> None:
    """SIGTERM the process group of every registered live child."""
    with _child_pids_lock:
        pids = list(_child_pids)
    for pid in pids:
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            continue


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
            spawn. Exceptions raised by the callback are suppressed. The
            child is registered for cleanup_children() regardless.

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

    register_child_pid(proc.pid)
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
        unregister_child_pid(proc.pid)

    return SubprocResult(
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )
