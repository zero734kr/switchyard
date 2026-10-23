"""Build the argv for each service, plus its environment.

Control services are long-lived singletons on the control node; workers are one
``dynamo.vllm`` process per configured worker. The dispatcher is a plain table
of builders rather than a chain of name comparisons.
"""

from ..paths import expand
from ..slurm import placement
from . import engine, environment


def etcd(launch, run_dir):
    control = launch.settings.control
    head = environment.resolve(launch.recipe.control_node)
    url = f"http://{head}:{control.etcd_port}"
    peer = f"http://{head}:{control.etcd_peer_port}"
    return [
        expand(launch.settings.software.etcd),
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
    ]


def nats(launch, run_dir):
    head = environment.resolve(launch.recipe.control_node)
    return [
        expand(launch.settings.software.nats),
        "-js",
        "-a",
        head,
        "-p",
        str(launch.settings.control.nats_port),
        "--store_dir",
        str(run_dir / "nats"),
    ]


def frontend(launch, run_dir):
    settings = launch.settings
    return [
        environment.python(launch),
        "-m",
        "dynamo.frontend",
        "--http-host",
        "0.0.0.0",
        "--http-port",
        str(settings.frontend.port),
        "--router-mode",
        settings.frontend.router_mode,
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
    ]


def worker(launch, worker_spec):
    return [environment.python(launch), *engine.args(launch, worker_spec)]


CONTROL_BUILDERS = {"etcd": etcd, "nats": nats, "frontend": frontend}


def build(launch, service, run_dir):
    """Return ``(argv, env)`` for one service, ready for ``os.execvpe``."""
    recipe = launch.recipe
    env = environment.base(launch, run_dir)
    if environment.is_control(service):
        return [
            *placement.control_prefix(recipe, service),
            *CONTROL_BUILDERS[service](launch, run_dir),
        ], env
    worker_spec = recipe.worker(service)
    allocation_devices = environment.visible_devices(env)
    environment.worker(launch, worker_spec, env, allocation_devices)
    environment.select_devices(env, worker_spec)
    prefix = placement.worker_prefix(launch, worker_spec, allocation_devices)
    return [*prefix, *worker(launch, worker_spec)], env
