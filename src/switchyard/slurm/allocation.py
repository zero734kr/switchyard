"""Verify that the configured job IDs are running, owned by us, and contain the nodes.

Switchyard only ever *uses* existing allocations. It never submits, resizes or
cancels a job, so an expired allocation has to fail with a clear message rather
than silently degrading a run.
"""

import os
import subprocess
from dataclasses import dataclass

from ..errors import AllocationError
from ..paths import ROOT  # noqa: F401  (re-exported by callers to fill --chdir)


@dataclass(frozen=True)
class Allocation:
    job_id: int
    state: str
    owner: str
    nodes: tuple[str, ...]

    @property
    def running(self):
        return self.state in {"RUNNING", "COMPLETING"}


def _scontrol(*arguments):
    try:
        return subprocess.check_output(["scontrol", *arguments], text=True)
    except FileNotFoundError as error:
        raise AllocationError(
            "scontrol not found; run on a Jarvis login or compute node"
        ) from error
    except subprocess.CalledProcessError as error:
        raise AllocationError(
            f"scontrol {' '.join(arguments)} failed: {error.output.strip()}"
        ) from error


def show(job_id):
    """Parse one `scontrol show job -o` record."""
    record = _scontrol("show", "job", "-o", str(job_id))
    fields = dict(item.split("=", 1) for item in record.split() if "=" in item)
    nodes = tuple(_scontrol("show", "hostnames", fields.get("NodeList", "")).split())
    return Allocation(
        job_id=int(fields.get("JobId", job_id)),
        state=fields.get("JobState", "UNKNOWN"),
        owner=fields.get("UserId", "").split("(")[0],
        nodes=nodes,
    )


def validate(recipe):
    """Every declared allocation must be running, owned by us, and cover its node."""
    user = os.environ.get("USER", "")
    checked = {}
    for declared in recipe.allocations.values():
        allocation = show(declared.job)
        checked[declared.name] = allocation
        if not allocation.running or allocation.owner != user:
            raise AllocationError(
                f"{declared.job}: allocation is not RUNNING and owned by this user"
            )
        if declared.node not in allocation.nodes:
            raise AllocationError(
                f"{declared.job}: allocation does not contain {declared.node}"
            )
    return checked
