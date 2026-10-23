"""Sample live NIXL KV bytes and RDMA link traffic on the configured P/D nodes.

The NIXL payload rate comes from the decode workers' cumulative vLLM histogram
counter. RDMA port counters include every process on those host ports, not only
Switchyard traffic. Link rates are nominal active-port ceilings, never measured
application goodput.
"""

import json
import math
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
from urllib.request import urlopen

from ..paths import expand
from ..slurm.step import interpreter

METRICS = {
    "bytes": "vllm:nixl_bytes_transferred_sum",
    "xfer_seconds": "vllm:nixl_xfer_time_seconds_sum",
    "failed": "vllm:nixl_num_failed_transfers_total",
}

#: Runs on the remote node through srun; emits one JSON document.
REMOTE_SAMPLE = r"""
import glob, json, re, sys, time

duration = float(sys.argv[1])

def read(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return None

def snapshot():
    ports = []
    for base in sorted(glob.glob('/sys/class/infiniband/*/ports/*')):
        device, port = base.split('/')[-3], base.split('/')[-1]
        state_text = read(base + '/state') or 'unknown'
        state = state_text.split()[-1].upper()
        rate_text = read(base + '/rate') or ''
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*Gb/sec', rate_text)
        try:
            rate_gbps = float(match.group(1)) if match else None
        except (ValueError, AttributeError):
            rate_gbps = None
        def counter(name):
            value = read(base + '/counters/' + name)
            try:
                # Linux InfiniBand sysfs data counters are in 4-byte units.
                return int(value) * 4 if value is not None else None
            except ValueError:
                return None
        ports.append({
            'device': device, 'port': port, 'state': state,
            'rate_gbps': rate_gbps, 'rate_text': rate_text,
            'tx_bytes': counter('port_xmit_data'),
            'rx_bytes': counter('port_rcv_data'),
        })
    return ports

before = snapshot()
started = time.monotonic()
time.sleep(duration)
after = snapshot()
print(json.dumps({
    'before': before, 'after': after,
    'elapsed_s': time.monotonic() - started,
}), flush=True)
"""


def metric_sum(text, metric_name):
    """Sum one Prometheus sample family, ignoring labels and comments."""
    total = 0.0
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(None, 1)[0]
        if metric != metric_name:
            continue
        try:
            value = float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
        if math.isfinite(value):
            total += value
            count += 1
    return total if count else None


def scrape_worker(worker):
    url = f"http://{worker.node}:{worker.system_port}/metrics"
    try:
        with urlopen(url, timeout=3) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, URLError, TimeoutError) as error:
        return {"error": str(error), **{name: None for name in METRICS}}
    values = {name: metric_sum(text, metric) for name, metric in METRICS.items()}
    if values["bytes"] is None:
        values["error"] = "NIXL byte histogram not exposed at /metrics"
    return values


def scrape_decoders(workers):
    # Switchyard configures NIXL pull mode: the kv_consumer records each
    # transfer. Summing producer and consumer would double-count wherever both
    # sides expose the same operation.
    return {
        worker.name: scrape_worker(worker)
        for worker in workers
        if worker.role == "decode"
    }


def metric_delta(before, after, field):
    total = 0.0
    usable = 0
    issues = []
    for name in sorted(set(before) | set(after)):
        left = before.get(name, {}).get(field)
        right = after.get(name, {}).get(field)
        error = after.get(name, {}).get("error") or before.get(name, {}).get("error")
        if left is None or right is None:
            issues.append(f"{name}: {error or 'counter unavailable'}")
            continue
        delta = right - left
        if delta < 0:
            issues.append(f"{name}: counter reset during sample window")
            continue
        total += delta
        usable += 1
    return (total if usable else None), issues


def sample_node(job, node, python, window):
    command = [
        "srun",
        f"--jobid={job}",
        "--cpu-bind=none",
        "--overlap",
        "--exact",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        f"--nodelist={node}",
        "--gres=none",
        python,
        "-u",
        "-c",
        REMOTE_SAMPLE,
        str(window),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=window + 60
    )
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"{node}: worker sampler returned no JSON data")


def delta_counter(first, last):
    if first is None or last is None or last < first:
        return None
    return last - first


def to_gbps(byte_count, seconds):
    if byte_count is None or seconds <= 0:
        return None
    return byte_count * 8 / seconds / 1e9


def summarize_ports(sample):
    before = {(port["device"], port["port"]): port for port in sample["before"]}
    after = {(port["device"], port["port"]): port for port in sample["after"]}
    ports = []
    for key in sorted(set(before) | set(after)):
        first, last = before.get(key, {}), after.get(key, {})
        tx = delta_counter(first.get("tx_bytes"), last.get("tx_bytes"))
        rx = delta_counter(first.get("rx_bytes"), last.get("rx_bytes"))
        ports.append(
            {
                "device": key[0],
                "port": key[1],
                "state": last.get("state", first.get("state", "unknown")),
                "rate_gbps": last.get("rate_gbps") or first.get("rate_gbps"),
                "tx_bytes": tx,
                "rx_bytes": rx,
                "tx_gbps": to_gbps(tx, sample["elapsed_s"]),
                "rx_gbps": to_gbps(rx, sample["elapsed_s"]),
            }
        )
    return ports


def active_capacity(ports):
    rates = [
        port["rate_gbps"]
        for port in ports
        if port["state"] == "ACTIVE" and port["rate_gbps"] is not None
    ]
    return sum(rates) if rates else None


def aggregate_capacity(nodes, rows):
    values = [rows[node]["active_capacity_gbps"] for node in nodes]
    return (
        sum(values) if values and all(value is not None for value in values) else None
    )


def aggregate_rate(ports, field):
    values = [port[field] for port in ports if port[field] is not None]
    return sum(values) if values else None


def _node_jobs(recipe, workers):
    """One allocation per node, for the node-wide NIC counters.

    Several workers may share a node across different allocations, so any
    allocation covering the node can run the sampler.
    """
    nodes = {}
    for worker in sorted(workers, key=lambda w: (w.node, w.allocation)):
        nodes.setdefault(worker.node, recipe.allocation_for(worker).job)
    return nodes


def collect(launch, window, fabric_gbps=None):
    """One sampling window: decode-side NIXL counters plus per-node NIC deltas."""
    recipe = launch.recipe
    workers = recipe.workers
    nodes = _node_jobs(recipe, workers)
    python = interpreter(launch.settings.software)
    if not shutil.which("srun"):
        raise RuntimeError("run this command on a Jarvis host where srun is available")

    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        network_jobs = {
            node: pool.submit(sample_node, job, node, python, window)
            for node, job in nodes.items()
        }
        before = scrape_decoders(workers)
        started = time.monotonic()
        time.sleep(window)
        after = scrape_decoders(workers)
        metric_elapsed = time.monotonic() - started
        network_samples = {
            node: future.result() for node, future in network_jobs.items()
        }

    kv_bytes, byte_issues = metric_delta(before, after, "bytes")
    xfer_seconds, time_issues = metric_delta(before, after, "xfer_seconds")
    failed, failure_issues = metric_delta(before, after, "failed")
    active_gbps = (
        kv_bytes * 8 / xfer_seconds / 1e9
        if kv_bytes is not None and xfer_seconds and xfer_seconds > 0
        else None
    )

    roles_by_node = {}
    for worker in workers:
        roles_by_node.setdefault(worker.node, set()).add(worker.role)
    rows = {}
    for node, sample in network_samples.items():
        ports = summarize_ports(sample)
        rows[node] = {
            "roles": sorted(roles_by_node[node]),
            "elapsed_s": sample["elapsed_s"],
            "ports": ports,
            "active_capacity_gbps": active_capacity(ports),
            "tx_bytes": sum(port["tx_bytes"] or 0 for port in ports),
            "rx_bytes": sum(port["rx_bytes"] or 0 for port in ports),
            "tx_gbps": aggregate_rate(ports, "tx_gbps"),
            "rx_gbps": aggregate_rate(ports, "rx_gbps"),
        }

    prefill_nodes = sorted({w.node for w in workers if w.role == "prefill"})
    decode_nodes = sorted({w.node for w in workers if w.role == "decode"})
    prefill_capacity = aggregate_capacity(prefill_nodes, rows)
    decode_capacity = aggregate_capacity(decode_nodes, rows)
    ceiling = None
    if prefill_capacity is not None and decode_capacity is not None:
        limits = [prefill_capacity, decode_capacity]
        if fabric_gbps is not None:
            limits.append(fabric_gbps)
        ceiling = min(limits)
    same_node = set(prefill_nodes) == set(decode_nodes) and len(prefill_nodes) == 1
    target = launch.settings.fabric.target_utilization

    return {
        "recipe": recipe.name,
        "window_seconds": window,
        "kv_payload": {
            "source": "decode-worker vllm:nixl_bytes_transferred_sum",
            "bytes": kv_bytes,
            "wall_gbps": to_gbps(kv_bytes, metric_elapsed),
            "active_transfer_gbps": active_gbps,
            "failed_transfers": failed,
            "issues": sorted(set(byte_issues + time_issues + failure_issues)),
        },
        "nodes": rows,
        "capacity": {
            "prefill_egress_nominal_gbps": prefill_capacity,
            "decode_ingress_nominal_gbps": decode_capacity,
            "shared_fabric_gbps": fabric_gbps,
            "end_to_end_nominal_ceiling_gbps": ceiling,
            "target_utilization": target,
            "target_gbps": ceiling * target if ceiling is not None else None,
            "fabric_ceiling_is_known": fabric_gbps is not None,
            "same_node_p_and_d": same_node,
        },
    }


def gbps(value):
    return "n/a" if value is None else f"{value:.2f} Gbps"


def gbps_to_gb_s(value):
    return "n/a" if value is None else f"{value / 8:.3f} GB/s"


def report(result):
    """Human-readable version of :func:`collect`."""
    print(
        f"Recipe: {result['recipe']}  |  sample window: {result['window_seconds']:.1f}s"
    )
    kv = result["kv_payload"]
    print("\nNIXL KV payload (decode consumer metrics)")
    if kv["bytes"] is None:
        print("  unavailable; check the workers' /metrics endpoints and NIXL metrics")
    else:
        print(f"  transferred: {kv['bytes'] / 1e9:.3f} GB")
        print(
            f"  wall-window rate: {gbps_to_gb_s(kv['wall_gbps'])} "
            f"({gbps(kv['wall_gbps'])})"
        )
        active = gbps_to_gb_s(kv["active_transfer_gbps"])
        print(f"  transfer-active aggregate rate: {active}")
        failed = kv["failed"] if kv["failed"] is not None else "n/a"
        print(f"  failed transfers in window: {failed}")
    for issue in kv["issues"]:
        print(f"  note: {issue}")

    print("\nRDMA ports (node totals; includes other users/processes on those hosts)")
    for node, row in result["nodes"].items():
        roles = "+".join(row["roles"])
        print(
            f"  {node} [{roles}]: TX {gbps_to_gb_s(row['tx_gbps'])}, "
            f"RX {gbps_to_gb_s(row['rx_gbps'])}, "
            f"nominal active-port ceiling {gbps(row['active_capacity_gbps'])}"
        )
        for port in row["ports"]:
            print(
                f"    {port['device']}:{port['port']} {port['state']} "
                f"{gbps(port['rate_gbps'])}; TX {gbps_to_gb_s(port['tx_gbps'])}, "
                f"RX {gbps_to_gb_s(port['rx_gbps'])}"
            )

    capacity = result["capacity"]
    print("\nDoubleword fabric check: min(P egress, D ingress, shared fabric)")
    print(f"  prefill egress nominal: {gbps(capacity['prefill_egress_nominal_gbps'])}")
    print(f"  decode ingress nominal: {gbps(capacity['decode_ingress_nominal_gbps'])}")
    if capacity["shared_fabric_gbps"] is None:
        print("  shared fabric ceiling: unknown; pass --fabric-gbps if known")
    else:
        print(f"  shared fabric ceiling: {gbps(capacity['shared_fabric_gbps'])}")
    ceiling = gbps(capacity["end_to_end_nominal_ceiling_gbps"])
    print(f"  end-to-end nominal ceiling: {ceiling}")
    print(
        f"  configured target ({capacity['target_utilization']:.0%}): "
        f"{gbps(capacity['target_gbps'])}"
    )
    if capacity["same_node_p_and_d"]:
        print(
            "  P and D are on the same node; this recipe does not exercise "
            "cross-node fabric transfer."
        )
    elif not capacity["fabric_ceiling_is_known"]:
        print(
            "  This is an endpoint NIC ceiling only; switch capacity/oversubscription "
            "is not discovered here."
        )
    if kv["wall_gbps"] is not None and capacity["end_to_end_nominal_ceiling_gbps"]:
        ratio = kv["wall_gbps"] / capacity["end_to_end_nominal_ceiling_gbps"] * 100
        print(
            f"  NIXL payload / nominal ceiling: {ratio:.2f}% "
            "(wire utilization is shown by NIC counters)"
        )
    print(
        "  NIC TX/RX counters are host-wide RDMA traffic, "
        "not process-isolated KV bytes."
    )
    print(
        "  The nominal link ceiling is not measured RDMA goodput; use a separate "
        "fabric benchmark for that."
    )


__all__ = ["METRICS", "collect", "expand", "report"]
