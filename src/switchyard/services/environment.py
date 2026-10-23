"""Environment handed to a service, and the GPU/UCX policy applied per role."""

import os
import socket
import subprocess

from ..analysis import proximity
from ..config import CONTROL_SERVICES
from ..paths import KERNELS, python_in

#: Variables that must not leak from the launching shell into a service.
INHERITED_PREFIXES = ("JARVIS_",)
INHERITED_NAMES = ("SWITCHYARD_CUSTOM_KERNELS",)
#: Dynamo 1.5 interprets a response-stream host override as an interface name,
#: so keep its working auto-selection unless transport.environment overrides it.
STALE_DYNAMO_NAMES = ("DYN_TCP_RESPONSE_STREAM_HOST", "DYN_EVENT_PLANE_HOST")
#: Let UCX discover node-local transports and every available HCA. Inherited
#: pins from an earlier experiment are a common source of silent slowdowns.
UCX_AUTO = {
    "UCX_NET_DEVICES": "all",
    "UCX_TLS": "all",
    "UCX_MAX_RNDV_RAILS": "2",
}


def _clean():
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(INHERITED_PREFIXES) and name not in INHERITED_NAMES
    }
    for name in STALE_DYNAMO_NAMES:
        env.pop(name, None)
    return env


def host_ip():
    """This node's address, as Dynamo and NIXL will advertise it."""
    return socket.gethostbyname(socket.gethostname())


def resolve(name):
    return socket.gethostbyname(name)


def base(launch, run_dir):
    """Environment shared by every service in a run."""
    settings = launch.settings
    host = host_ip()
    head = resolve(launch.recipe.control_node)
    control = settings.control
    env = _clean()
    env.update(
        ETCD_ENDPOINTS=f"http://{head}:{control.etcd_port}",
        NATS_SERVER=f"nats://{head}:{control.nats_port}",
        DYN_DISCOVERY_BACKEND="etcd",
        DYN_REQUEST_PLANE="tcp",
        DYN_EVENT_PLANE="nats",
        DYN_TCP_RPC_HOST=host,
        DYN_NAMESPACE="switchyard-" + run_dir.name,
    )
    env.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
    env.update(
        {key: str(value) for key, value in settings.transport.environment.items()}
    )
    return env


def _topo_matrix():
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proximity.parse(result.stdout)


def select_hca(launch, worker_spec, env):
    """Pin UCX to the HCA attached to this worker's GPU, when one exists.

    ``fabric.hca_selection: local`` is opt-in. A cross-socket path is never
    pinned: UCX's own choice across both cards is at least as good, and pinning
    would hide the fact that no local HCA exists.
    """
    settings = launch.settings
    if settings.fabric.hca_selection != "local":
        return
    if "UCX_NET_DEVICES" in settings.transport.environment:
        print(
            f"Switchyard fabric {worker_spec.name}: keeping explicit "
            f"UCX_NET_DEVICES={settings.transport.environment['UCX_NET_DEVICES']}",
            flush=True,
        )
        return
    ordinals = (
        worker_spec.gpu_ids
        if worker_spec.gpu_ids is not None
        else tuple(range(worker_spec.tp))
    )
    devices, grade = proximity.best_hcas(_topo_matrix(), ordinals)
    if devices and proximity.pinnable(grade):
        env["UCX_NET_DEVICES"] = ",".join(devices)
        print(
            f"Switchyard fabric {worker_spec.name}: UCX_NET_DEVICES="
            f"{env['UCX_NET_DEVICES']} ({grade} to GPU {','.join(map(str, ordinals))})",
            flush=True,
        )
        return
    print(
        f"Switchyard fabric {worker_spec.name}: no locally attached HCA "
        f"(best {grade or 'unknown'}); leaving UCX device selection to the runtime",
        flush=True,
    )


def _apply_transport_overrides(env, settings, variables):
    env.update(variables)
    for name in variables:
        if name in settings.transport.environment:
            env[name] = str(settings.transport.environment[name])


def visible_devices(env):
    exposed = env.get("CUDA_VISIBLE_DEVICES", "")
    return [device.strip() for device in exposed.split(",") if device.strip()]


def select_devices(env, worker):
    """Apply ``gpu_ids`` to Slurm's ``CUDA_VISIBLE_DEVICES`` list."""
    devices = visible_devices(env)
    if worker.gpu_ids is not None:
        if not devices:
            raise ValueError(
                f"Slurm did not expose CUDA_VISIBLE_DEVICES for {worker.name}; "
                "cannot map gpu_ids"
            )
        if any(gpu_id >= len(devices) for gpu_id in worker.gpu_ids):
            raise ValueError(
                f"{worker.name}.gpu_ids={list(worker.gpu_ids)} exceeds the "
                f"{len(devices)} GPUs exposed by Slurm"
            )
        selected = [devices[gpu_id] for gpu_id in worker.gpu_ids]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
        selected_text = env["CUDA_VISIBLE_DEVICES"]
        print(
            f"Switchyard GPU assignment {worker.name}: allocation-visible indices "
            f"{list(worker.gpu_ids)} -> CUDA_VISIBLE_DEVICES={selected_text}",
            flush=True,
        )
    visible = visible_devices(env)
    if len(visible) < worker.tp:
        raise ValueError(f"Slurm has not exposed {worker.tp} GPUs for {worker.name}")
    return visible


def worker(launch, worker_spec, env, visible):
    """Add the per-worker runtime variables (ports, NIXL, kernels, UCX)."""
    settings = launch.settings
    _apply_transport_overrides(env, settings, dict(UCX_AUTO))
    select_hca(launch, worker_spec, env)
    print(
        f"Switchyard UCX policy {worker_spec.name}: "
        f"NET_DEVICES={env['UCX_NET_DEVICES']} TLS={env['UCX_TLS']} "
        f"MAX_RNDV_RAILS={env['UCX_MAX_RNDV_RAILS']}",
        flush=True,
    )
    host = host_ip()
    env.update(
        DYN_SYSTEM_PORT=str(worker_spec.system_port),
        VLLM_NIXL_SIDE_CHANNEL_HOST=host,
        VLLM_NIXL_SIDE_CHANNEL_PORT=str(worker_spec.nixl_port),
        FLASHINFER_MAX_NUM_TOKENS=str(settings.engine.max_num_batched_tokens),
        VLLM_SSM_CONV_STATE_LAYOUT="DS",
    )
    if settings.engine.kernels == "custom":
        env["PYTHONPATH"] = str(KERNELS) + os.pathsep + env.get("PYTHONPATH", "")
        env["SWITCHYARD_CUSTOM_KERNELS"] = launch.model_key
    return env


def is_control(service):
    return service in CONTROL_SERVICES


def python(launch):
    return python_in(launch.settings.software.venv)


__all__ = [
    "base",
    "is_control",
    "python",
    "select_devices",
    "select_hca",
    "visible_devices",
    "worker",
]
