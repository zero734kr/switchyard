"""Choose CPU and memory affinity inside a Slurm allocation on this node.

Placement is best effort but never silent: every fallback prints a warning and
an explicit ``numa_node`` request that cannot be honored is an error.
"""

import ctypes
import os
import shutil
from pathlib import Path

from ..errors import PlacementError

#: A worker with no GPUs pinned still gets a balanced share of the pool.
DEFAULT_GPUS_PER_WORKER = 1


def ids(value):
    """Parse a Linux cpulist such as ``0-3,8,10-11``."""
    result = set()
    for span in value.strip().split(","):
        if span:
            low, _, high = span.partition("-")
            result.update(range(int(low), int(high or low) + 1))
    return result


def topology(allowed):
    """NUMA node -> CPUs and the list of physical cores, restricted to ``allowed``."""
    nodes = {}
    for path in Path("/sys/devices/system/node").glob("node[0-9]*"):
        nodes[int(path.name[4:])] = ids((path / "cpulist").read_text()) & allowed
    cores = set()
    for cpu in sorted(allowed):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        siblings = ids(path.read_text()) if path.exists() else {cpu}
        cores.add(tuple(sorted(siblings & allowed)))
    return nodes, sorted(cores)


def gpu_nodes(cuda_home, count):
    """NUMA node per CUDA ordinal.

    CUDA ordinals honor Slurm's visibility/remapping, unlike NVML indices, so
    the runtime API is the only correct source here.
    """
    runtime = None
    for library in (
        "libcudart.so",
        str(Path(cuda_home) / "targets/x86_64-linux/lib/libcudart.so"),
        str(Path(cuda_home) / "lib64/libcudart.so"),
    ):
        try:
            runtime = ctypes.CDLL(library)
            break
        except OSError:
            pass
    if runtime is None:
        raise PlacementError(
            "CUDA runtime library unavailable for GPU locality discovery"
        )
    query = runtime.cudaDeviceGetPCIBusId
    query.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    result = {}
    for ordinal in range(count):
        bus = ctypes.create_string_buffer(64)
        status = query(bus, len(bus), ordinal)
        if status:
            raise PlacementError(
                f"CUDA PCI lookup failed for ordinal {ordinal}: {status}"
            )
        domain, bus_id, device = bus.value.decode().lower().rsplit(":", 2)
        pci = f"{int(domain, 16):04x}:{bus_id}:{device}"
        path = Path("/sys/bus/pci/devices") / pci / "numa_node"
        node = int(path.read_text()) if path.exists() else -1
        result[ordinal] = {node} if node >= 0 else set()
    return result


def choose(workers, nodes, cores, preferred, reserve_control=False):
    """Partition physical cores; SMT siblings never go to different workers."""
    free = list(cores)
    if reserve_control:
        reserved = min(3, max(0, len(free) - len(workers)))
        if not reserved:
            print(
                "Switchyard placement warning: no spare physical core for control "
                "services; CPU sharing is unavoidable",
                flush=True,
            )
        del free[:reserved]
    assigned = {worker.name: [] for worker in workers}
    budgets = {worker.name: worker.cpus_per_task for worker in workers}

    def count(name):
        return len(assigned[name])

    def fits(name):
        return budgets[name] is None or count(name) < budgets[name]

    def local(name, core):
        return not preferred[name] or any(
            core[0] in nodes.get(node, set()) for node in preferred[name]
        )

    def distance(name, core):
        target = next((n for n, cpus in nodes.items() if core[0] in cpus), None)
        values = []
        for node in preferred[name]:
            path = Path(f"/sys/devices/system/node/node{node}/distance")
            if target is not None and path.exists():
                row = list(map(int, path.read_text().split()))
                index = sorted(nodes).index(target)
                if index < len(row):
                    values.append(row[index])
        return min(values, default=100)

    # Give constrained workers first choice; at least two physical cores when
    # available, then distribute remaining local cores fairly.
    order = sorted(
        workers,
        key=lambda worker: (
            sum(local(worker.name, core) for core in free),
            worker.name,
        ),
    )
    for _ in range(2):
        for worker in order:
            name = worker.name
            candidates = [
                core for core in free if worker.numa_node is None or local(name, core)
            ]
            if candidates and fits(name):
                core = min(
                    candidates, key=lambda c: (not local(name, c), distance(name, c), c)
                )
                assigned[name].append(core)
                free.remove(core)
    for core in free:
        candidates = [
            worker.name
            for worker in workers
            if fits(worker.name) and local(worker.name, core)
        ]
        if candidates:
            name = min(candidates, key=lambda n: (count(n), n))
            assigned[name].append(core)
    result = {}
    for worker in workers:
        name = worker.name
        if not assigned[name]:
            print(
                "Switchyard placement warning: insufficient physical cores; "
                f"{name} must share",
                flush=True,
            )
            candidates = [
                core for core in cores if worker.numa_node is None or local(name, core)
            ]
            assigned[name] = [candidates[0]]
        # With a CPU cap, use separate physical cores before SMT siblings.
        cpus = [core[0] for core in assigned[name]]
        cpus += [cpu for core in assigned[name] for cpu in core[1:]]
        result[name] = cpus[: budgets[name]] if budgets[name] is not None else cpus
    return result


def prefix(cpus, memory_nodes, name):
    """Wrap a command so it runs on ``cpus`` with memory bound to ``memory_nodes``."""
    cpu_list = ",".join(map(str, sorted(cpus)))
    if shutil.which("numactl"):
        memory = (
            "--membind=" + ",".join(map(str, sorted(memory_nodes)))
            if memory_nodes
            else "--localalloc"
        )
        print(
            f"Switchyard placement {name}: cpus={cpu_list} memory={memory}", flush=True
        )
        return ["numactl", f"--physcpubind={cpu_list}", memory]
    print(
        f"Switchyard placement warning: numactl unavailable; {name} gets CPU affinity "
        "and default first-touch memory",
        flush=True,
    )
    os.sched_setaffinity(0, set(cpus))
    return []


def control_prefix(recipe, name):
    """Keep etcd, NATS and the frontend on separate physical cores."""
    _, cores = topology(set(os.sched_getaffinity(0)))
    control_allocation = recipe.control_allocation().name
    workers = sum(worker.allocation == control_allocation for worker in recipe.workers)
    count = max(1, min(3, len(cores) - workers))
    index = ("etcd", "nats", "frontend").index(name) % count
    return prefix(cores[index], set(), name)


def worker_prefix(launch, worker, visible):
    """CPU/memory binding for one worker, preferring its GPU's NUMA node."""
    settings, recipe = launch.settings, launch.recipe
    allowed = set(os.sched_getaffinity(0))
    nodes, cores = topology(allowed)
    peers = recipe.peers(worker)
    try:
        locality = gpu_nodes(settings.software.cuda_home, len(visible))
    except (OSError, PlacementError) as error:
        print(
            f"Switchyard placement warning: {error}; using balanced CPU affinity "
            "without inferred GPU locality",
            flush=True,
        )
        locality = {}
    preferred = {}
    for peer in peers:
        explicit = peer.numa_node
        if explicit is not None:
            if not nodes.get(explicit):
                raise PlacementError(
                    f"{peer.name}: NUMA node {explicit} has no CPUs in this "
                    "Slurm allocation"
                )
            preferred[peer.name] = {explicit}
        else:
            ordinals = peer.gpu_ids or list(range(peer.tp))
            preferred[peer.name] = set().union(
                *(locality.get(i, set()) for i in ordinals)
            )
    placements = choose(
        peers,
        nodes,
        cores,
        preferred,
        worker.allocation == recipe.control_allocation().name,
    )
    name = worker.name
    cpus = placements[name]
    if not preferred[name]:
        print(
            f"Switchyard placement warning: GPU NUMA locality is unknown for {name}; "
            "using balanced allocation CPUs",
            flush=True,
        )
    if preferred[name] and not any(
        cpu in nodes.get(node, set()) for cpu in cpus for node in preferred[name]
    ):
        print(
            f"Switchyard placement warning: {name} has no allocated CPU near its GPU; "
            "using nearest available CPUs",
            flush=True,
        )
    mems = next(
        line.split(":", 1)[1]
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Mems_allowed_list:")
    )
    memory = preferred[name] & ids(mems)
    if worker.numa_node is not None and not memory:
        raise PlacementError(
            f"{name}: requested NUMA memory node is outside the Slurm allocation"
        )
    if worker.numa_node is not None and not shutil.which("numactl"):
        raise PlacementError(f"{name}: explicit NUMA memory binding requires numactl")
    if preferred[name] and not memory:
        print(
            f"Switchyard placement warning: {name} cannot bind GPU-local memory "
            "in this allocation; using first-touch memory",
            flush=True,
        )
    return prefix(cpus, memory, name)
