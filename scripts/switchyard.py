"""Prepare or run cross-node Dynamo using existing user-owned Slurm allocations."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.request

from config import DEFAULT_CONFIG, ROOT, expand, read_config, resolve


def worker_args(c, model, w):
    e = c["engine"]
    args = [
        "-m",
        "dynamo.vllm",
        "--model",
        expand(model["path"]),
        "--served-model-name",
        model["served_name"],
        "--tensor-parallel-size",
        str(w["tp"]),
        "--dtype",
        e["dtype"],
        "--gpu-memory-utilization",
        str(e["gpu_memory_utilization"]),
        "--max-model-len",
        str(e["max_model_len"]),
        "--max-num-seqs",
        str(e["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(e["max_num_batched_tokens"]),
        "--discovery-backend",
        "etcd",
        "--disaggregation-mode",
        w["role"],
        "--language-model-only",
        "--enable-chunked-prefill",
        "--reasoning-parser",
        "qwen3",
        "--gdn-prefill-backend",
        e["gdn_prefill_backend"],
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_producer" if w["role"] == "prefill" else "kv_consumer",
                "kv_load_failure_policy": "fail",
            }
        ),
    ]
    if e["prefix_caching"]:
        args.append("--enable-prefix-caching")
    if e["mtp_tokens"]:
        args += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": e["mtp_tokens"]}),
        ]
    if w["role"] == "prefill":
        args += [
            "--kv-events-config",
            json.dumps(
                {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "endpoint": f"tcp://*:{w['event_port']}",
                    "enable_kv_cache_events": True,
                }
            ),
        ]
    return args


def service(c, r, model, key, name, run_dir):
    """Resolve a command on its destination node; Slurm owns visible GPU IDs."""
    host = socket.gethostbyname(socket.gethostname())
    head = socket.gethostbyname(r["control_node"])
    ctrl = c["control"]
    env = dict(os.environ)
    # Dynamo 1.5 interprets the response-stream host override as an interface.
    # Keep its working auto-selection unless transport.environment overrides it.
    env.pop("DYN_TCP_RESPONSE_STREAM_HOST", None)
    env.pop("DYN_EVENT_PLANE_HOST", None)
    for variable in list(env):
        if variable.startswith("JARVIS_") or variable == "SWITCHYARD_CUSTOM_KERNELS":
            env.pop(variable)
    env.update(
        ETCD_ENDPOINTS=f"http://{head}:{ctrl['etcd_port']}",
        NATS_SERVER=f"nats://{head}:{ctrl['nats_port']}",
        DYN_DISCOVERY_BACKEND="etcd",
        DYN_REQUEST_PLANE="tcp",
        DYN_EVENT_PLANE="nats",
        DYN_TCP_RPC_HOST=host,
        DYN_NAMESPACE="switchyard-" + run_dir.name,
    )
    env.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
    env.update({k: str(v) for k, v in c["transport"]["environment"].items()})
    py = expand(c["software"]["venv"]) + "/bin/python"
    if name == "etcd":
        url = f"http://{head}:{ctrl['etcd_port']}"
        peer = f"http://{head}:{ctrl['etcd_peer_port']}"
        return [
            expand(c["software"]["etcd"]),
            "--name",
            "switchyard",
            "--data-dir",
            str(run_dir / "etcd"),
            "--listen-client-urls",
            url,
            "--advertise-client-urls",
            url,
            "--listen-peer-urls",
            peer,
            "--initial-advertise-peer-urls",
            peer,
            "--initial-cluster",
            "switchyard=" + peer,
        ], env
    if name == "nats":
        return [
            expand(c["software"]["nats"]),
            "-js",
            "-a",
            head,
            "-p",
            str(ctrl["nats_port"]),
            "--store_dir",
            str(run_dir / "nats"),
        ], env
    if name == "frontend":
        return [
            py,
            "-m",
            "dynamo.frontend",
            "--http-host",
            "0.0.0.0",
            "--http-port",
            str(c["frontend"]["port"]),
            "--router-mode",
            c["frontend"]["router_mode"],
            "--discovery-backend",
            "etcd",
            "--dyn-chat-processor",
            "vllm",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--default-chat-template-kwargs",
            '{"enable_thinking":true,"preserve_thinking":true}',
        ], env
    w = next(w for w in r["workers"] if w["name"] == name)
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) < w["tp"]:
        raise ValueError(f"Slurm has not exposed {w['tp']} GPUs for {name}")
    env.update(
        DYN_SYSTEM_PORT=str(w["system_port"]),
        VLLM_NIXL_SIDE_CHANNEL_HOST=host,
        VLLM_NIXL_SIDE_CHANNEL_PORT=str(w["nixl_port"]),
        FLASHINFER_MAX_NUM_TOKENS=str(c["engine"]["max_num_batched_tokens"]),
        VLLM_SSM_CONV_STATE_LAYOUT="DS",
    )
    if c["engine"]["kernels"] == "custom":
        env["PYTHONPATH"] = (
            str(ROOT / "kernels") + os.pathsep + env.get("PYTHONPATH", "")
        )
        env["SWITCHYARD_CUSTOM_KERNELS"] = key
    return [py, *worker_args(c, model, w)], env


def step(c, r, a, name, run_dir):
    node = next(
        (w["node"] for w in r["workers"] if w["name"] == name), r["control_node"]
    )
    setup = [
        "set -euo pipefail",
        "cd " + shlex.quote(str(ROOT)),
        "source " + shlex.quote(str(ROOT / "recipes/jarvis.sh")),
        "switchyard_runtime "
        + shlex.join([c["software"]["cuda_home"], *c["software"]["modules"]]),
        "exec "
        + shlex.join(
            [
                expand(c["software"]["venv"]) + "/bin/python",
                str(Path(__file__).resolve()),
                "role",
                a.recipe,
                "--config",
                str(a.config.resolve()),
                "--model",
                a.model or r["model"],
                "--service",
                name,
                "--run-dir",
                str(run_dir),
            ]
        ),
    ]
    return [
        "srun",
        f"--jobid={r['allocations'][node]}",
        "--overlap",
        "--exact",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        f"--nodelist={node}",
        "bash",
        "-lc",
        "\n".join(setup),
    ]


def validate_allocations(r):
    for node, job in r["allocations"].items():
        record = subprocess.check_output(
            ["scontrol", "show", "job", "-o", str(job)], text=True
        )
        fields = dict(item.split("=", 1) for item in record.split() if "=" in item)
        if (
            fields.get("JobState") != "RUNNING"
            or fields.get("UserId", "").split("(")[0] != os.environ["USER"]
        ):
            raise ValueError(f"{job}: allocation is not RUNNING and owned by this user")
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", fields["NodeList"]], text=True
        ).split()
        if node not in nodes:
            raise ValueError(f"{job}: allocation does not contain {node}")


def validate_runtime(c):
    py = expand(c["software"]["venv"]) + "/bin/python"
    probe = "import importlib.metadata as m,json,sys; print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))"
    versions = json.loads(
        subprocess.check_output(
            [py, "-c", probe, *c["software"]["versions"]], text=True
        )
    )
    for package, expected in c["software"]["versions"].items():
        actual = versions[package]
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, installed {actual}")


def run(c, r, model, key, a, directory, check_only=False):
    validate_allocations(r)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(a.config.read_text())
    a.config = directory / "config.yaml"
    processes, logs = [], []

    def start(name):
        log = (directory / f"{name}.log").open("w")
        logs.append(log)
        cmd = step(c, r, a, name, directory)
        processes.append(
            subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        )
        print(f"started {name}; log={log.name}", flush=True)

    def alive():
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("a service exited; inspect run logs")

    def wait(url, model_check=False):
        deadline = time.monotonic() + min(
            c["control"]["startup_timeout"],
            180 if check_only else c["control"]["startup_timeout"],
        )
        while time.monotonic() < deadline:
            alive()
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if not model_check:
                        return
                    data = json.load(response)
                if any(item["id"] == model["served_name"] for item in data.get("data", [])):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError(url)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        start("etcd")
        wait(f"http://{r['control_node']}:{c['control']['etcd_port']}/health")
        start("nats")
        # Check NATS protocol readiness, not only an open socket.
        deadline = time.monotonic() + 30
        while True:
            alive()
            try:
                with socket.create_connection(
                    (r["control_node"], c["control"]["nats_port"]), timeout=2
                ) as sock:
                    if not sock.recv(4096).startswith(b"INFO "):
                        raise OSError("not a NATS server")
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("NATS startup")
                time.sleep(1)
        if check_only:
            start("frontend")
            wait(
                f"http://{r['control_node']}:{c['frontend']['port']}/v1/models"
            )
            probe = "\n".join(
                [
                    "import socket,urllib.request,json",
                    f"assert json.load(urllib.request.urlopen('http://{r['control_node']}:{c['control']['etcd_port']}/health',timeout=5))['health'] in ('true', True)",
                    f"s=socket.create_connection(('{r['control_node']}',{c['control']['nats_port']}),timeout=5)",
                    "assert s.recv(4096).startswith(b'INFO '); s.close()",
                    f"assert isinstance(json.load(urllib.request.urlopen('http://{r['control_node']}:{c['frontend']['port']}/v1/models',timeout=5))['data'], list)",
                    "print(socket.gethostname(), 'etcd + NATS + Dynamo frontend reachable; CPU-only setup check')",
                ]
            )
            for node, job in r["allocations"].items():
                subprocess.run(
                    [
                        "srun",
                        f"--jobid={job}",
                        "--overlap",
                        "--exact",
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=1",
                        f"--nodelist={node}",
                        expand(c["software"]["venv"]) + "/bin/python",
                        "-c",
                        probe,
                    ],
                    check=True,
                    timeout=45,
                )
                alive()
            return
        start("frontend")
        for w in r["workers"]:
            start(w["name"])
        wait(
            f"http://{r['control_node']}:{c['frontend']['port']}/v1/models",
            True,
        )
        wait(f"http://{r['control_node']}:{c['frontend']['port']}/health")
        print(
            f"ready at http://{r['control_node']}:{c['frontend']['port']}/v1; Ctrl-C preserves salloc allocations",
            flush=True,
        )
        while True:
            alive()
            time.sleep(2)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for p in reversed(processes):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in processes:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                p.wait()
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("plan", "preflight", "control-check", "run", "role")
    )
    p.add_argument("recipe", nargs="?", default="initial-h100")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=("dense", "moe"))
    p.add_argument("--service")
    p.add_argument("--run-dir", type=Path)
    a = p.parse_args()
    c = read_config(a.config)
    r, model, key = resolve(c, a.recipe, a.model)
    directory = (
        a.run_dir
        or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ).resolve()
    if a.action == "role":
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("roles must run in a Slurm job step")
        if a.service not in {
            "etcd",
            "nats",
            "frontend",
            *(w["name"] for w in r["workers"]),
        }:
            raise ValueError("role requires a valid --service")
        node = next(
            (w["node"] for w in r["workers"] if w["name"] == a.service),
            r["control_node"],
        )
        if (
            socket.gethostname().split(".")[0] != node
            or int(os.environ["SLURM_JOB_ID"]) != r["allocations"][node]
        ):
            raise ValueError("role placement does not match the configured allocation")
        validate_runtime(c)
        os.environ.update(
            SWITCHYARD_CONFIG=str(a.config.resolve()), SWITCHYARD_RECIPE=a.recipe
        )
        cmd, env = service(c, r, model, key, a.service, directory)
        os.execvpe(cmd[0], cmd, env)
    elif a.action == "preflight":
        validate_allocations(r)
        validate_runtime(c)
        missing = [
            expand(c["software"][k])
            for k in ("etcd", "nats")
            if not Path(expand(c["software"][k])).is_file()
        ]
        if not Path(expand(model["path"])).is_dir():
            missing.append(expand(model["path"]))
        if missing:
            raise ValueError(f"missing dependencies: {missing}")
        print(
            "allocation ownership/placement, config, model and control binaries checked; no GPUs initialized"
        )
    elif a.action == "plan":
        for name in [
            "etcd",
            "nats",
            "frontend",
            *(w["name"] for w in r["workers"]),
        ]:
            print(shlex.join(step(c, r, a, name, directory)))
        for w in r["workers"]:
            print(f"# {w['name']} engine args: {shlex.join(worker_args(c, model, w))}")
    else:
        run(c, r, model, key, a, directory, check_only=a.action == "control-check")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
