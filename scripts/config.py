"""One YAML configuration for Dynamo serving and Slurm tooling."""

import argparse
import json
import os
from pathlib import Path
import shlex
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def read_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ValueError("config.yaml requires schema_version: 2")
    return config


def resolve(config, name, model=None):
    recipe = config["recipes"][name]
    model = model or recipe["model"]
    if model not in config["models"]:
        raise ValueError(f"unknown model: {model}")
    nodes = recipe["allocations"]
    if recipe["control_node"] not in nodes or not recipe["workers"]:
        raise ValueError(
            f"{name}: configure control_node, allocations and workers first"
        )
    if not all(
        isinstance(job, int) and not isinstance(job, bool) and job > 0
        for job in nodes.values()
    ):
        raise ValueError("allocation IDs must be positive integers")
    workers = recipe["workers"]
    if not any(w["role"] == "prefill" for w in workers) or not any(
        w["role"] == "decode" for w in workers
    ):
        raise ValueError("a recipe needs prefill and decode workers")
    if len({w["name"] for w in workers}) != len(workers):
        raise ValueError("worker names must be unique")
    if any(w["name"] in ("etcd", "nats", "frontend") for w in workers):
        raise ValueError("worker names must not shadow control services")
    if len({w["tp"] for w in workers}) != 1:
        raise ValueError("the pinned hybrid-model connector requires matching P/D TP")
    ports = {}
    control = recipe["control_node"]
    for key in ("etcd_port", "etcd_peer_port", "nats_port"):
        ports.setdefault(control, []).append(config["control"][key])
    ports[control].append(config["frontend"]["port"])
    if config["frontend"]["router_mode"] not in ("kv", "round-robin"):
        raise ValueError("frontend.router_mode must be kv or round-robin")
    workers_by_node = {}
    for w in workers:
        if w["node"] not in nodes or w["role"] not in ("prefill", "decode"):
            raise ValueError(f"invalid worker placement: {w}")
        if not isinstance(w["tp"], int) or isinstance(w["tp"], bool) or w["tp"] < 1:
            raise ValueError("tp must be a positive integer")
        gpu_ids = w.get("gpu_ids")
        if gpu_ids is not None and (
            not isinstance(gpu_ids, list)
            or len(gpu_ids) != w["tp"]
            or any(
                not isinstance(gpu_id, int)
                or isinstance(gpu_id, bool)
                or gpu_id < 0
                for gpu_id in gpu_ids
            )
            or len(set(gpu_ids)) != len(gpu_ids)
        ):
            raise ValueError(
                f"{w['name']}.gpu_ids must contain tp unique nonnegative integers"
            )
        workers_by_node.setdefault(w["node"], []).append(w)
        ports.setdefault(w["node"], []).extend(
            w[k] for k in ("system_port", "nixl_port", "event_port") if k in w
        )
    for node, colocated in workers_by_node.items():
        if len(colocated) < 2:
            continue
        if any(w.get("gpu_ids") is None for w in colocated):
            raise ValueError(
                f"workers sharing {node} must each set gpu_ids to avoid sharing devices"
            )
        gpu_ids = [gpu_id for w in colocated for gpu_id in w["gpu_ids"]]
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"workers sharing {node} must use disjoint gpu_ids")
    for node, values in ports.items():
        if len(values) != len(set(values)) or any(
            not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= 65535
            for p in values
        ):
            raise ValueError(f"invalid or colliding ports on {node}")
    e = config["engine"]
    for key in (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "mtp_tokens",
    ):
        if (
            not isinstance(e[key], int)
            or isinstance(e[key], bool)
            or e[key] < (0 if key == "mtp_tokens" else 1)
        ):
            raise ValueError(f"invalid engine.{key}")
    if not 0 < e["gpu_memory_utilization"] <= 1 or e["kernels"] not in (
        "baseline",
        "custom",
    ):
        raise ValueError("invalid memory utilization or kernels selection")
    if not 0 < config["fabric"]["target_utilization"] <= 1:
        raise ValueError("invalid fabric.target_utilization")
    return recipe, config["models"][model], model


def expand(value):
    return os.path.expanduser(os.path.expandvars(str(value)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", nargs="?", default="initial-h100")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=("dense", "moe"))
    p.add_argument("--check", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    c = read_config(a.config)
    if a.list:
        for name, r in c["recipes"].items():
            print(name, "configured" if r["allocations"] else "awaiting allocations")
        return
    r, m, _ = resolve(c, a.recipe, a.model)
    if a.check:
        print(
            f"{a.recipe}: valid; MTP={c['engine']['mtp_tokens']}, prefix cache={c['engine']['prefix_caching']}, kernels={c['engine']['kernels']}"
        )
        return
    values = {
        "PROFILE": a.recipe,
        "EXPECTED_NODE": r["control_node"],
        "DYNAMO_VENV": expand(c["software"]["venv"]),
        "MODEL_PATH": expand(m["path"]),
        "MODEL_NAME": m["served_name"],
        "CUDA_HOME": c["software"]["cuda_home"],
        "GCC_MODULE": c["software"]["modules"][0],
        "CUDA_MODULE": c["software"]["modules"][1],
        "MAX_NUM_BATCHED_TOKENS": str(c["engine"]["max_num_batched_tokens"]),
    }
    if a.json:
        print(json.dumps(values, indent=2))
    else:
        for key, value in values.items():
            print(f"export {key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
