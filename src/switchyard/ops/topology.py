"""Read-only topology report: what placement and fabric selection will do.

Runs one srun step per allocation because Slurm's cgroup hides the other
allocation's GPUs: a single step can only report the pair it owns.
"""

import base64
import json
import shlex
import subprocess
from dataclasses import dataclass

from ..analysis import proximity

#: Runs on the compute node and prints one JSON document.
REMOTE = r"""
import glob, json, os, subprocess

def read(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return None

def run(*command):
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        return ""

QUERY = ("nvidia-smi", "--query-gpu=index,pci.bus_id,name", "--format=csv,noheader")

gpus = []
for line in run(*QUERY).splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 3:
        continue
    index, bdf, name = fields
    pci = bdf.lower().replace("00000000:", "0000:")
    gpus.append({
        "index": int(index),
        "bdf": bdf,
        "name": name,
        "numa": read("/sys/bus/pci/devices/" + pci + "/numa_node"),
    })

hcas = []
for path in sorted(glob.glob("/sys/class/infiniband/*")):
    hcas.append({
        "device": os.path.basename(path),
        "state": read(path + "/ports/1/state"),
        "rate": read(path + "/ports/1/rate"),
    })

status = {}
for line in (read("/proc/self/status") or "").splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        status[key] = value.strip()

print(json.dumps({
    "host": run("hostname", "-s").strip(),
    "cpus": status.get("Cpus_allowed_list"),
    "mems": status.get("Mems_allowed_list"),
    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpus": gpus,
    "hcas": hcas,
    "topo": run("nvidia-smi", "topo", "-m"),
    "numactl": run("numactl", "-H"),
}))
"""


@dataclass(frozen=True)
class NodeView:
    """What one allocation's step could see."""

    allocation: str
    job: int
    node: str
    host: str
    cpus: str
    mems: str
    gpus: tuple[dict, ...]
    hcas: tuple[dict, ...]
    matrix: proximity.Matrix | None

    def gpu(self, ordinal):
        return next((gpu for gpu in self.gpus if gpu["index"] == ordinal), None)


def command(settings, allocation):
    """The srun command that collects one allocation's view."""
    # Standard library only: the inspector must work on a CPU-only control node
    # that has no inference virtual environment.
    payload = base64.b64encode(REMOTE.encode()).decode()
    script = f"echo {payload} | base64 -d | python3 -"
    return [
        "srun",
        f"--jobid={allocation.job}",
        "--cpu-bind=none",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        f"--nodelist={allocation.node}",
        "bash",
        "-lc",
        script,
    ]


def _parse_json(output):
    for line in reversed(output.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("topology probe returned no JSON")


def inspect(launch):
    """Collect one :class:`NodeView` per allocation, in recipe order."""
    settings = launch.settings
    views = []
    for allocation in launch.recipe.allocations.values():
        result = subprocess.run(
            command(settings, allocation),
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        data = _parse_json(result.stdout)
        views.append(
            NodeView(
                allocation=allocation.name,
                job=allocation.job,
                node=allocation.node,
                host=data["host"],
                cpus=data.get("cpus") or "?",
                mems=data.get("mems") or "?",
                gpus=tuple(data.get("gpus") or ()),
                hcas=tuple(data.get("hcas") or ()),
                matrix=proximity.parse(data.get("topo") or ""),
            )
        )
    return views


def _ordinals(worker):
    return worker.gpu_ids if worker.gpu_ids is not None else tuple(range(worker.tp))


def _fabric_line(view, worker, select_local):
    devices, grade = proximity.best_hcas(view.matrix, _ordinals(worker))
    if not devices:
        return "unknown", "UCX chooses"
    graded = ",".join(devices)
    if not select_local:
        return f"{graded} ({grade})", "all (selection off)"
    if proximity.pinnable(grade):
        return f"{graded} ({grade})", graded
    return f"{graded or 'cross-socket'} ({grade})", "all (no local HCA)"


def report(launch, views, verbose=True):
    """Print the resolved per-worker picture and return it as data."""
    recipe = launch.recipe
    select_local = launch.settings.fabric.hca_selection == "local"
    by_allocation = {view.allocation: view for view in views}
    rows = []
    for worker in recipe.workers:
        allocation = recipe.allocation_for(worker)
        view = by_allocation[allocation.name]
        ordinals = _ordinals(worker)
        gpus = [view.gpu(ordinal) for ordinal in ordinals]
        gpus = [gpu for gpu in gpus if gpu]
        numa = sorted({gpu["numa"] for gpu in gpus if gpu["numa"] not in (None, "-1")})
        attached, net = _fabric_line(view, worker, select_local)
        rows.append(
            {
                "worker": worker.name,
                "role": worker.role,
                "allocation": allocation.name,
                "job": allocation.job,
                "node": allocation.node,
                "gpus": [gpu["bdf"] for gpu in gpus],
                "numa": numa,
                "best_hca": attached,
                "ucx_net_devices": net,
                "cpus": view.cpus,
                "mems": view.mems,
            }
        )

    if not verbose:
        return {"workers": rows, "hca_selection": launch.settings.fabric.hca_selection}

    width = max(len(row["worker"]) for row in rows)
    print(
        f"{'worker':<{width}} {'node':<6} {'alloc':<8} {'GPUs':<28} "
        f"{'numa':<6} {'best HCA':<18} UCX_NET_DEVICES"
    )
    for row in rows:
        print(
            f"{row['worker']:<{width}} {row['node']:<6} {row['allocation']:<8} "
            f"{','.join(row['gpus']) or '-':<28} "
            f"{','.join(row['numa']) or '-':<6} {row['best_hca']:<18} "
            f"{row['ucx_net_devices']}"
        )

    carriers = sorted({hca["device"] for view in views for hca in view.hcas})
    print(f"\nHCAs on these nodes: {', '.join(carriers) or 'none'}")
    for view in views:
        states = "; ".join(
            f"{hca['device']} {hca['state'] or '?'} {hca['rate'] or ''}".strip()
            for hca in view.hcas
        )
        print(f"  {view.node} ({view.allocation}): {states or 'no HCAs'}")

    warnings = _warnings(rows, select_local)
    if warnings:
        print("\nnotes")
        for warning in warnings:
            print(f"  - {warning}")
    return {"workers": rows, "hca_selection": launch.settings.fabric.hca_selection}


def _warnings(rows, select_local):
    messages = []
    if not select_local:
        messages.append(
            "fabric.hca_selection is 'all'; UCX picks a device per connection"
        )
    cross = [row["worker"] for row in rows if "SYS" in row["best_hca"]]
    if cross:
        messages.append(
            "no locally attached HCA for "
            + ", ".join(cross)
            + " (transfers cross the CPU interconnect either way)"
        )
    for row in rows:
        if len(row["numa"]) > 1:
            messages.append(
                f"{row['worker']} spans NUMA nodes {', '.join(row['numa'])}"
            )
    return messages


def shell_command(launch, allocation):
    """Exposed for ``--show-commands`` and for reproducing a probe by hand."""
    return shlex.join(command(launch.settings, allocation))
