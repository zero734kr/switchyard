"""The launcher actions: plan, preflight, control-check, run, and the ``role``
entry point those steps re-exec into on the target node.
"""

import json
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

from .config import CONTROL_SERVICES
from .errors import ConfigError
from .paths import expand
from .services import commands, engine, supervisor
from .slurm import allocation, build_step, interpreter, probe_command

CHECK_ONLY_CAP = 180.0


def validate_runtime(settings):
    """The inference venv must be exactly the validated stack."""
    python = interpreter(settings.software)
    probe = (
        "import importlib.metadata as m,json,sys;"
        " print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))"
    )
    versions = json.loads(
        subprocess.check_output(
            [python, "-c", probe, *settings.software.versions], text=True
        )
    )
    for package, expected in settings.software.versions.items():
        actual = versions[package]
        if actual != expected:
            raise ConfigError(f"{package}: expected {expected}, installed {actual}")
    return versions


def _services(recipe):
    return ["etcd", "nats", "frontend", *recipe.worker_names]


def _step(launch, service, run_dir, config_path):
    return build_step(
        launch,
        service,
        run_dir,
        config_path,
        launch.model_key,
        launch.recipe.name,
    )


def plan(launch, run_dir, config_path):
    """Print the exact commands a run would execute."""
    for service in _services(launch.recipe):
        print(shlex.join(_step(launch, service, run_dir, config_path)))
    for worker in launch.recipe.workers:
        print(f"# {worker.name} engine args: {shlex.join(engine.args(launch, worker))}")
        if worker.gpu_ids is not None:
            print(
                f"# {worker.name} CUDA_VISIBLE_DEVICES selects allocation-visible "
                "indices " + ",".join(map(str, worker.gpu_ids))
            )


def preflight(launch):
    """Everything that can be checked without initializing a GPU."""
    settings = launch.settings
    allocation.validate(launch.recipe)
    validate_runtime(settings)
    missing = [
        expand(getattr(settings.software, name))
        for name in ("etcd", "nats")
        if not Path(expand(getattr(settings.software, name))).is_file()
    ]
    if not Path(expand(launch.model.path)).is_dir():
        missing.append(expand(launch.model.path))
    if missing:
        raise ConfigError(f"missing dependencies: {missing}")
    print(
        "allocation ownership/placement, config, model and control binaries checked; "
        "no GPUs initialized"
    )


def role(launch, service, run_dir, config_path):
    """Re-exec into one service inside a Slurm step on the target node."""
    recipe = launch.recipe
    if not os.environ.get("SLURM_JOB_ID"):
        raise ConfigError("roles must run in a Slurm job step")
    if service not in _services(recipe):
        raise ConfigError("role requires a valid --service")
    target = (
        recipe.control_allocation()
        if service in CONTROL_SERVICES
        else recipe.allocation_for(recipe.worker(service))
    )
    if (
        socket.gethostname().split(".")[0] != target.node
        or int(os.environ["SLURM_JOB_ID"]) != target.job
    ):
        raise ConfigError(
            f"role placement does not match allocation {target.name} "
            f"({target.node}, job {target.job})"
        )
    validate_runtime(launch.settings)
    os.environ.update(
        SWITCHYARD_CONFIG=str(Path(config_path).resolve()),
        SWITCHYARD_RECIPE=recipe.name,
    )
    command, env = commands.build(launch, service, run_dir)
    os.execvpe(command[0], command, env)


def _probe_script(launch):
    control = launch.settings.control
    frontend = launch.settings.frontend
    head = launch.recipe.control_node
    return "\n".join(
        [
            "import socket,urllib.request,json",
            f"assert json.load(urllib.request.urlopen("
            f"'http://{head}:{control.etcd_port}/health',timeout=5))['health'] "
            "in ('true', True)",
            f"s=socket.create_connection(('{head}',{control.nats_port}),timeout=5)",
            "assert s.recv(4096).startswith(b'INFO '); s.close()",
            f"assert isinstance(json.load(urllib.request.urlopen("
            f"'http://{head}:{frontend.port}/v1/models',timeout=5))['data'], list)",
            "print(socket.gethostname(), 'etcd + NATS + Dynamo frontend reachable; "
            "CPU-only setup check')",
        ]
    )


def _check_only(launch, monitor, directory, config_path):
    """Validate the control plane from every allocation, without loading a model."""
    recipe = launch.recipe
    frontend = launch.settings.frontend
    monitor.start(_step(launch, "frontend", directory, config_path), "frontend")
    monitor.wait(
        f"http://{recipe.control_node}:{frontend.port}/v1/models", cap=CHECK_ONLY_CAP
    )
    probe = _probe_script(launch)
    # Probe every allocation: two allocations on one node are two independent
    # steps, and a wrong job ID only shows up in the one that is broken.
    for declared in recipe.allocations.values():
        subprocess.run(probe_command(launch, declared, probe), check=True, timeout=45)
        monitor.check()


def run(launch, directory, check_only=False):
    """Start every service, wait for readiness, then supervise until Ctrl-C."""
    recipe = launch.recipe
    settings = launch.settings
    directory = Path(directory).resolve()
    allocation.validate(recipe)
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "config.yaml"
    config_path.write_text(settings.path.read_text())

    supervisor.install_sigterm_handler()
    monitor = supervisor.Supervisor(directory, settings.control.startup_timeout)
    control_node = socket.gethostbyname(recipe.control_node)
    try:
        monitor.start(_step(launch, "etcd", directory, config_path), "etcd")
        monitor.wait(f"http://{control_node}:{settings.control.etcd_port}/health")
        monitor.start(_step(launch, "nats", directory, config_path), "nats")
        monitor.wait_nats(control_node, settings.control.nats_port)
        if check_only:
            _check_only(launch, monitor, directory, config_path)
            return
        monitor.start(_step(launch, "frontend", directory, config_path), "frontend")
        for worker in recipe.workers:
            monitor.start(
                _step(launch, worker.name, directory, config_path), worker.name
            )
        endpoint = f"http://{recipe.control_node}:{settings.frontend.port}"
        monitor.wait(
            f"{endpoint}/v1/models",
            model_check=True,
            served_name=launch.model.served_name,
        )
        monitor.wait(f"{endpoint}/health")
        print(
            f"ready at {endpoint}/v1; Ctrl-C preserves salloc allocations", flush=True
        )
        while True:
            monitor.check()
            time.sleep(supervisor.READINESS_INTERVAL)
    finally:
        monitor.stop()
