#!/usr/bin/env python3
"""Sample live NIXL KV bytes and RDMA link traffic on configured P/D workers.

The NIXL payload rate comes from the decode workers' cumulative vLLM histogram
counter. RDMA port counters include all traffic on those host ports, not only
Switchyard traffic. Link rates are nominal active-port ceilings, not measured
application goodput.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from config import DEFAULT_CONFIG, expand, read_config, resolve


METRICS = {
    "bytes": "vllm:nixl_bytes_transferred_sum",
    "xfer_seconds": "vllm:nixl_xfer_time_seconds_sum",
    "failed": "vllm:nixl_num_failed_transfers_total",
}

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
    url = f"http://{worker['node']}:{worker['system_port']}/metrics"
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
    # transfer. Summing producer and consumer would double-count where both
    # sides expose the same operation.
    return {w["name"]: scrape_worker(w) for w in workers if w["role"] == "decode"}


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


def summarize_ports(sample):
    before = {(p["device"], p["port"]): p for p in sample["before"]}
    after = {(p["device"], p["port"]): p for p in sample["after"]}
    ports = []
    for key in sorted(set(before) | set(after)):
        first, last = before.get(key, {}), after.get(key, {})
        tx = delta_counter(first.get("tx_bytes"), last.get("tx_bytes"))
        rx = delta_counter(first.get("rx_bytes"), last.get("rx_bytes"))
        rate = last.get("rate_gbps") or first.get("rate_gbps")
        ports.append({
            "device": key[0], "port": key[1],
            "state": last.get("state", first.get("state", "unknown")),
            "rate_gbps": rate, "tx_bytes": tx, "rx_bytes": rx,
            "tx_gbps": to_gbps(tx, sample["elapsed_s"]),
            "rx_gbps": to_gbps(rx, sample["elapsed_s"]),
        })
    return ports


def delta_counter(first, last):
    if first is None or last is None or last < first:
        return None
    return last - first


def to_gbps(byte_count, seconds):
    if byte_count is None or seconds <= 0:
        return None
    return byte_count * 8 / seconds / 1e9


def active_capacity(ports):
    rates = [
        p["rate_gbps"]
        for p in ports
        if p["state"] == "ACTIVE" and p["rate_gbps"] is not None
    ]
    return sum(rates) if rates else None


def aggregate_capacity(nodes, node_rows):
    values = [node_rows[node]["active_capacity_gbps"] for node in nodes]
    return sum(values) if values and all(value is not None for value in values) else None


def aggregate_rate(ports, field):
    values = [port[field] for port in ports if port[field] is not None]
    return sum(values) if values else None


def gbps(value):
    return "n/a" if value is None else f"{value:.2f} Gbps"


def gbps_to_gb_s(value):
    return "n/a" if value is None else f"{value / 8:.3f} GB/s"


def collect(c, recipe, workers, window, fabric_gbps):
    nodes = {}
    for worker in workers:
        node = worker["node"]
        job = recipe["allocations"][node]
        if node in nodes and nodes[node] != job:
            raise ValueError(f"workers on {node} refer to different allocations")
        nodes[node] = job

    python = expand(c["software"]["venv"]) + "/bin/python"
    if not shutil.which("srun"):
        raise RuntimeError("Run this script on a Jarvis host where srun is available")

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
        network_samples = {node: future.result() for node, future in network_jobs.items()}

    kv_bytes, byte_issues = metric_delta(before, after, "bytes")
    xfer_seconds, time_issues = metric_delta(before, after, "xfer_seconds")
    failed, failure_issues = metric_delta(before, after, "failed")
    kv_gbps = to_gbps(kv_bytes, metric_elapsed)
    active_gbps = (
        kv_bytes * 8 / xfer_seconds / 1e9
        if kv_bytes is not None and xfer_seconds and xfer_seconds > 0
        else None
    )

    roles_by_node = {}
    for worker in workers:
        roles_by_node.setdefault(worker["node"], set()).add(worker["role"])
    node_rows = {}
    for node, sample in network_samples.items():
        ports = summarize_ports(sample)
        node_rows[node] = {
            "roles": sorted(roles_by_node[node]),
            "elapsed_s": sample["elapsed_s"],
            "ports": ports,
            "active_capacity_gbps": active_capacity(ports),
            "tx_bytes": sum(p["tx_bytes"] or 0 for p in ports),
            "rx_bytes": sum(p["rx_bytes"] or 0 for p in ports),
            "tx_gbps": aggregate_rate(ports, "tx_gbps"),
            "rx_gbps": aggregate_rate(ports, "rx_gbps"),
        }

    prefill_nodes = sorted({w["node"] for w in workers if w["role"] == "prefill"})
    decode_nodes = sorted({w["node"] for w in workers if w["role"] == "decode"})
    prefill_cap = aggregate_capacity(prefill_nodes, node_rows)
    decode_cap = aggregate_capacity(decode_nodes, node_rows)
    ceiling = None
    if prefill_cap is not None and decode_cap is not None:
        limits = [prefill_cap, decode_cap]
        if fabric_gbps is not None:
            limits.append(fabric_gbps)
        ceiling = min(limits)
    same_node = set(prefill_nodes) == set(decode_nodes) and len(prefill_nodes) == 1

    return {
        "recipe": recipe["name"],
        "window_seconds": window,
        "kv_payload": {
            "source": "decode-worker vllm:nixl_bytes_transferred_sum",
            "bytes": kv_bytes,
            "wall_gbps": kv_gbps,
            "active_transfer_gbps": active_gbps,
            "failed_transfers": failed,
            "issues": sorted(set(byte_issues + time_issues + failure_issues)),
        },
        "nodes": node_rows,
        "capacity": {
            "prefill_egress_nominal_gbps": prefill_cap,
            "decode_ingress_nominal_gbps": decode_cap,
            "shared_fabric_gbps": fabric_gbps,
            "end_to_end_nominal_ceiling_gbps": ceiling,
            "target_utilization": c["fabric"]["target_utilization"],
            "target_gbps": ceiling * c["fabric"]["target_utilization"] if ceiling is not None else None,
            "fabric_ceiling_is_known": fabric_gbps is not None,
            "same_node_p_and_d": same_node,
        },
    }


def print_report(result, as_json=False):
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"Recipe: {result['recipe']}  |  sample window: {result['window_seconds']:.1f}s")
    kv = result["kv_payload"]
    print("\nNIXL KV payload (decode consumer metrics)")
    if kv["bytes"] is None:
        print("  unavailable; check the workers' /metrics endpoints and NIXL metrics")
    else:
        print(f"  transferred: {kv['bytes'] / 1e9:.3f} GB")
        print(f"  wall-window rate: {gbps_to_gb_s(kv['wall_gbps'])} ({gbps(kv['wall_gbps'])})")
        print(f"  transfer-active aggregate rate: {gbps_to_gb_s(kv['active_transfer_gbps'])}")
        print(f"  failed transfers in window: {kv['failed'] if kv['failed'] is not None else 'n/a'}")
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

    cap = result["capacity"]
    print("\nDoubleword fabric check: min(P egress, D ingress, shared fabric)")
    print(f"  prefill egress nominal: {gbps(cap['prefill_egress_nominal_gbps'])}")
    print(f"  decode ingress nominal: {gbps(cap['decode_ingress_nominal_gbps'])}")
    if cap["shared_fabric_gbps"] is None:
        print("  shared fabric ceiling: unknown; pass --fabric-gbps if known")
    else:
        print(f"  shared fabric ceiling: {gbps(cap['shared_fabric_gbps'])}")
    print(f"  end-to-end nominal ceiling: {gbps(cap['end_to_end_nominal_ceiling_gbps'])}")
    print(f"  configured target ({cap['target_utilization']:.0%}): {gbps(cap['target_gbps'])}")
    if cap["same_node_p_and_d"]:
        print("  P and D are on the same node; this recipe does not exercise cross-node fabric transfer.")
    elif not cap["fabric_ceiling_is_known"]:
        print("  This is an endpoint NIC ceiling only; switch capacity/oversubscription is not discovered here.")
    if kv["wall_gbps"] is not None and cap["end_to_end_nominal_ceiling_gbps"]:
        ratio = kv["wall_gbps"] / cap["end_to_end_nominal_ceiling_gbps"] * 100
        print(f"  NIXL payload / nominal ceiling: {ratio:.2f}% (wire utilization is shown by NIC counters)")
    print("  NIC TX/RX counters are host-wide RDMA traffic, not process-isolated KV bytes.")
    print("  The nominal link ceiling is not measured RDMA goodput; use a separate fabric benchmark for that.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", nargs="?", default="initial-h100")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("dense", "moe"))
    parser.add_argument("--window", type=float, default=10.0, help="measurement window in seconds")
    parser.add_argument("--fabric-gbps", type=float, help="known shared path capacity in Gbps")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.fabric_gbps is not None and args.fabric_gbps <= 0:
        parser.error("--fabric-gbps must be positive")
    config = read_config(args.config)
    try:
        recipe, _model, _key = resolve(config, args.recipe, args.model)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    recipe = {**recipe, "name": args.recipe}
    try:
        result = collect(config, recipe, recipe["workers"], args.window, args.fabric_gbps)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"fabric monitor failed: {error}", file=sys.stderr)
        return 1
    print_report(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
