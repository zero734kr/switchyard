"""Service process supervision: start, wait for readiness, clean up.

Workers are started in their own sessions so a Ctrl-C on the launcher can take
down the whole ``srun`` tree with one signal, while the Slurm allocations
themselves are left untouched.
"""

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from ..errors import LaunchError
from ..paths import ROOT

READINESS_INTERVAL = 2.0
NATS_TIMEOUT = 30.0
CLEANUP_TIMEOUT = 30.0


class Supervisor:
    """Owns the child processes of one run directory."""

    def __init__(self, directory, startup_timeout):
        self.directory = Path(directory)
        self.startup_timeout = startup_timeout
        self.processes = []
        self.logs = []

    def start(self, command, name):
        log = (self.directory / f"{name}.log").open("w")
        self.logs.append(log)
        self.processes.append(
            subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        print(f"started {name}; log={log.name}", flush=True)

    def check(self):
        if any(process.poll() is not None for process in self.processes):
            raise LaunchError(f"a service exited; inspect the logs in {self.directory}")

    def wait(self, url, model_check=False, served_name=None, cap=None):
        """Poll ``url`` until it answers (and lists ``served_name`` if asked)."""
        deadline = time.monotonic() + min(
            self.startup_timeout, cap if cap is not None else self.startup_timeout
        )
        while time.monotonic() < deadline:
            self.check()
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if not model_check:
                        return
                    data = json.load(response)
                if any(item["id"] == served_name for item in data.get("data", [])):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(READINESS_INTERVAL)
        raise LaunchError(f"timed out waiting for {url}")

    def wait_nats(self, host, port):
        """Check the NATS protocol greeting, not merely an open socket."""
        deadline = time.monotonic() + NATS_TIMEOUT
        while True:
            self.check()
            try:
                with socket.create_connection((host, port), timeout=2) as connection:
                    if not connection.recv(4096).startswith(b"INFO "):
                        raise OSError("not a NATS server")
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise LaunchError(
                        "timed out waiting for NATS to speak the protocol"
                    ) from None
                time.sleep(1)

    def stop(self):
        """Terminate children, escalate to SIGKILL, and always close the logs."""
        previous = {
            number: signal.signal(number, signal.SIG_IGN)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for process in reversed(self.processes):
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
            for process in self.processes:
                try:
                    process.wait(timeout=CLEANUP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for log in self.logs:
                log.close()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
        return False


def install_sigterm_handler():
    """Turn SIGTERM into KeyboardInterrupt so cleanup runs before exit."""

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)


def run_directory(base=None):
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return ((base or ROOT / "runs") / stamp).resolve()
