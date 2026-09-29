#!/usr/bin/env python3
"""Validate config.json and emit one experiment profile as shell exports."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import NoReturn


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.json"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"config: {message}")


def read_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        fail("schema_version must be 1")
    for key in ("software", "model", "engine", "profiles"):
        if not isinstance(config.get(key), dict):
            fail(f"{key} must be an object")
    return config


def require(profile: dict, key: str, expected_type: type):
    value = profile.get(key)
    if not isinstance(value, expected_type) or isinstance(value, bool):
        fail(f"profile field {key!r} must be {expected_type.__name__}")
    return value


def validate_profile(name: str, profile: dict) -> None:
    require(profile, "node", str)
    require(profile, "accelerator", str)
    topology = require(profile, "topology", str)
    require(profile, "frontend_port", int)
    ports = [profile["frontend_port"]]

    if topology != "pd":
        fail(f"{name}: only the 'pd' topology is supported")
    require(profile, "prefill_gpu", str)
    decode_gpus = require(profile, "decode_gpus", list)
    system_ports = require(profile, "decode_system_ports", list)
    nixl_ports = require(profile, "decode_nixl_ports", list)
    if not decode_gpus or not all(isinstance(item, str) for item in decode_gpus):
        fail(f"{name}: decode_gpus must contain GPU indices")
    if len(decode_gpus) != len(system_ports) or len(decode_gpus) != len(nixl_ports):
        fail(f"{name}: every decode worker needs one system and NIXL port")
    for key in ("prefill_system_port", "prefill_nixl_port", "kv_event_port"):
        ports.append(require(profile, key, int))
    if not all(isinstance(port, int) and not isinstance(port, bool) for port in system_ports + nixl_ports):
        fail(f"{name}: decode ports must be integers")
    ports.extend(system_ports + nixl_ports)

    if any(port < 1 or port > 65535 for port in ports):
        fail(f"{name}: ports must be between 1 and 65535")
    if len(ports) != len(set(ports)):
        fail(f"{name}: ports must be unique on a node")


def validate(config: dict) -> None:
    engine = config["engine"]
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        value = engine.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            fail(f"engine.{key} must be a positive integer")
    mtp_tokens = engine.get("mtp_tokens")
    if not isinstance(mtp_tokens, int) or isinstance(mtp_tokens, bool) or mtp_tokens < 0:
        fail("engine.mtp_tokens must be a non-negative integer")
    utilization = engine.get("gpu_memory_utilization")
    if not isinstance(utilization, (int, float)) or isinstance(utilization, bool) or not 0 < utilization <= 1:
        fail("engine.gpu_memory_utilization must be in (0, 1]")
    if not config["profiles"]:
        fail("at least one profile is required")
    for name, profile in config["profiles"].items():
        if not isinstance(profile, dict):
            fail(f"profile {name!r} must be an object")
        validate_profile(name, profile)
        if "max_model_len" in profile:
            value = profile["max_model_len"]
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= engine["max_model_len"]:
                fail(f"{name}: max_model_len must be a positive integer no larger than engine.max_model_len")
        if "max_num_seqs" in profile:
            value = profile["max_num_seqs"]
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= engine["max_num_seqs"]:
                fail(f"{name}: max_num_seqs must be a positive integer no larger than engine.max_num_seqs")

    gateway = config.get("gateway")
    if not isinstance(gateway, dict):
        fail("gateway must be an object")
    require(gateway, "host", str)
    gateway_port = require(gateway, "port", int)
    backends = require(gateway, "backends", list)
    if not 1 <= gateway_port <= 65535 or len(backends) < 2:
        fail("gateway needs a valid port and at least two backends")
    names = set()
    for backend in backends:
        if not isinstance(backend, dict):
            fail("each gateway backend must be an object")
        name = require(backend, "name", str)
        url = require(backend, "url", str)
        weight = backend.get("weight")
        if name in names or not url.startswith(("http://", "https://")):
            fail("gateway backends need unique names and HTTP URLs")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            fail(f"gateway backend {name!r} needs a positive weight")
        names.add(name)


def expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def environment(config: dict, name: str) -> dict[str, str]:
    try:
        profile = config["profiles"][name]
    except KeyError:
        fail(f"unknown profile {name!r}; choose from {', '.join(config['profiles'])}")

    software = config["software"]
    model = config["model"]
    engine = config["engine"]
    values = {
        "PROFILE": name,
        "EXPECTED_NODE": profile["node"],
        "ACCELERATOR": profile["accelerator"],
        "TOPOLOGY": profile["topology"],
        "DYNAMO_VENV": expand(str(software["venv"])),
        "GCC_MODULE": str(software["gcc_module"]),
        "CUDA_MODULE": str(software["cuda_module"]),
        "CUDA_HOME": expand(str(software["cuda_home"])),
        "MODEL_PATH": expand(str(model["path"])),
        "MODEL_NAME": str(model["served_name"]),
        "MODEL_DTYPE": str(model["dtype"]),
        "GPU_MEMORY_UTILIZATION": str(engine["gpu_memory_utilization"]),
        "MAX_MODEL_LEN": str(profile.get("max_model_len", engine["max_model_len"])),
        "MAX_NUM_SEQS": str(profile.get("max_num_seqs", engine["max_num_seqs"])),
        "MAX_NUM_BATCHED_TOKENS": str(engine["max_num_batched_tokens"]),
        "MTP_TOKENS": str(engine["mtp_tokens"]),
        "FRONTEND_PORT": str(profile["frontend_port"]),
    }
    values.update(
        PREFILL_GPU=profile["prefill_gpu"],
        DECODE_GPUS=",".join(profile["decode_gpus"]),
        PREFILL_SYSTEM_PORT=str(profile["prefill_system_port"]),
        DECODE_SYSTEM_PORTS=",".join(map(str, profile["decode_system_ports"])),
        PREFILL_NIXL_PORT=str(profile["prefill_nixl_port"]),
        DECODE_NIXL_PORTS=",".join(map(str, profile["decode_nixl_ports"])),
        KV_EVENT_PORT=str(profile["kv_event_port"]),
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit resolved JSON instead of shell")
    args = parser.parse_args()

    config = read_config(args.config)
    validate(config)
    if args.check:
        print(f"{args.config}: {len(config['profiles'])} profiles valid")
        return
    if args.list:
        for name, profile in config["profiles"].items():
            print(f"{name}\t{profile['node']}\t{profile['topology']}\t{profile['accelerator']}")
        return
    if not args.profile:
        parser.error("profile is required unless --check or --list is used")

    values = environment(config, args.profile)
    if args.json:
        json.dump(values, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        for key, value in values.items():
            print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
