"""Typed access to config.yaml: parse once, validate in one place, then read.

The YAML schema (``schema_version: 2``) is unchanged; this module replaces the
dict walking and the validation that used to be spread across the launcher.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .paths import DEFAULT_CONFIG

SCHEMA_VERSION = 2
CONTROL_SERVICES = ("etcd", "nats", "frontend")
ROLES = ("prefill", "decode")
ROUTER_MODES = ("kv", "round-robin")
HCA_SELECTION = ("all", "local")
KERNEL_MODES = ("baseline", "custom")


@dataclass(frozen=True)
class Software:
    """Pinned runtimes. Nothing here is installed by this repository."""

    venv: str
    modules: tuple[str, ...]
    cuda_home: str
    versions: Mapping[str, str]
    etcd: str
    nats: str
    control_versions: Mapping[str, str]


@dataclass(frozen=True)
class Model:
    key: str
    path: str
    served_name: str


@dataclass(frozen=True)
class Engine:
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    mtp_tokens: int
    prefix_caching: bool
    gdn_prefill_backend: str
    kernels: str
    enforce_eager: bool = False


@dataclass(frozen=True)
class Frontend:
    port: int
    router_mode: str


@dataclass(frozen=True)
class Control:
    etcd_port: int
    etcd_peer_port: int
    nats_port: int
    startup_timeout: float

    def listeners(self):
        return (self.etcd_port, self.etcd_peer_port, self.nats_port)


@dataclass(frozen=True)
class Transport:
    environment: Mapping[str, str]


@dataclass(frozen=True)
class Fabric:
    target_utilization: float
    kv_bytes_per_element: float
    recurrent_bytes_per_element: float
    conv_bytes_per_element: float
    hca_selection: str = "all"


@dataclass(frozen=True)
class Allocation:
    """One existing Slurm allocation: a job ID covering exactly one node.

    Several allocations may cover the same node; each is an independent GPU and
    CPU pool, exposed to its own steps through Slurm's CUDA_VISIBLE_DEVICES.
    """

    name: str
    job: int
    node: str


@dataclass(frozen=True)
class Worker:
    name: str
    role: str
    allocation: str
    node: str
    tp: int
    system_port: int
    nixl_port: int
    event_port: int | None = None
    gpu_ids: tuple[int, ...] | None = None
    cpus_per_task: int | None = None
    numa_node: int | None = None


@dataclass(frozen=True)
class Recipe:
    name: str
    model: str
    control_node: str
    allocations: Mapping[str, Allocation]
    workers: tuple[Worker, ...]
    control_allocation_name: str | None = None

    @property
    def worker_names(self):
        return tuple(worker.name for worker in self.workers)

    def worker(self, name):
        for worker in self.workers:
            if worker.name == name:
                return worker
        raise ConfigError(f"{self.name}: unknown worker {name!r}")

    def allocation_for(self, worker):
        return self.allocations[worker.allocation]

    def control_allocation(self):
        """The allocation hosting etcd, NATS and the frontend."""
        if self.control_allocation_name is not None:
            return self.allocations[self.control_allocation_name]
        covering = [a for a in self.allocations.values() if a.node == self.control_node]
        # The shorthand key ``control_node: <job>`` names the allocation after
        # its node; prefer that one when several allocations cover the node.
        named = next((a for a in covering if a.name == self.control_node), None)
        return named or sorted(covering, key=lambda a: a.name)[0]

    def node_of(self, service):
        """Node hosting a control service or a named worker."""
        worker = next((w for w in self.workers if w.name == service), None)
        return worker.node if worker else self.control_node

    def peers(self, worker):
        """Workers sharing this worker's *allocation*, ordered by name.

        Placement partitions one allocation's CPU pool, so peers stop at the
        allocation boundary even when two allocations sit on the same node.
        """
        return tuple(
            sorted(
                (w for w in self.workers if w.allocation == worker.allocation),
                key=lambda w: w.name,
            )
        )


@dataclass(frozen=True)
class Settings:
    """The whole file, parsed.

    Recipes stay raw until something asks for one: a recipe may deliberately be
    a placeholder that only becomes valid once suitable allocations exist, and
    ``--list`` has to be able to report that without failing.
    """

    path: Path
    software: Software
    models: Mapping[str, Model]
    engine: Engine
    frontend: Frontend
    control: Control
    transport: Transport
    fabric: Fabric
    recipes: Mapping[str, Mapping[str, Any]]

    def recipe(self, name):
        """Validate and build one recipe on demand."""
        try:
            raw = self.recipes[name]
        except KeyError:
            available = ", ".join(sorted(self.recipes)) or "none"
            raise ConfigError(
                f"unknown recipe {name!r}; available: {available}"
            ) from None
        return _recipe(name, raw, self.models, self.control, self.frontend)

    def configured(self, name):
        """True when a recipe carries allocations; placeholder recipes do not."""
        return bool(self.recipes[name].get("allocations"))


@dataclass(frozen=True)
class Launch:
    """Everything one command needs: settings, the chosen recipe, its model."""

    settings: Settings
    recipe: Recipe
    model: Model

    @property
    def model_key(self):
        return self.model.key


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a list")
    return value


def _integer(value, name, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(value, name, minimum=0):
    return None if value is None else _integer(value, name, minimum)


def _fraction(value, message, low=0.0):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not low < value <= 1
    ):
        raise ConfigError(message)
    return float(value)


def _read(path):
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"{path} requires schema_version: {SCHEMA_VERSION}")
    return raw


def _software(raw):
    raw = _mapping(raw, "software")
    return Software(
        venv=str(raw.get("venv")),
        modules=tuple(str(module) for module in raw.get("modules", ())),
        cuda_home=str(raw.get("cuda_home")),
        versions=dict(raw.get("versions", {})),
        etcd=str(raw.get("etcd")),
        nats=str(raw.get("nats")),
        control_versions=dict(raw.get("control_versions", {})),
    )


def _models(raw):
    return {
        key: Model(
            key=key, path=str(value["path"]), served_name=str(value["served_name"])
        )
        for key, value in _mapping(raw, "models").items()
    }


def _engine(raw):
    raw = _mapping(raw, "engine")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        _integer(raw.get(key), f"engine.{key}", 1)
    if raw.get("kernels") not in KERNEL_MODES:
        raise ConfigError("invalid memory utilization or kernels selection")
    return Engine(
        dtype=str(raw.get("dtype", "bfloat16")),
        gpu_memory_utilization=_fraction(
            raw.get("gpu_memory_utilization"),
            "invalid memory utilization or kernels selection",
        ),
        max_model_len=raw["max_model_len"],
        max_num_seqs=raw["max_num_seqs"],
        max_num_batched_tokens=raw["max_num_batched_tokens"],
        mtp_tokens=_integer(raw.get("mtp_tokens", 0), "engine.mtp_tokens", 0),
        prefix_caching=bool(raw.get("prefix_caching", False)),
        gdn_prefill_backend=str(raw.get("gdn_prefill_backend", "flashinfer")),
        kernels=raw["kernels"],
        enforce_eager=bool(raw.get("enforce_eager", False)),
    )


def _frontend(raw):
    raw = _mapping(raw, "frontend")
    if raw.get("router_mode") not in ROUTER_MODES:
        raise ConfigError("frontend.router_mode must be kv or round-robin")
    return Frontend(
        port=_integer(raw.get("port"), "frontend.port", 1),
        router_mode=raw["router_mode"],
    )


def _control(raw):
    raw = _mapping(raw, "control")
    return Control(
        etcd_port=_integer(raw.get("etcd_port"), "control.etcd_port", 1),
        etcd_peer_port=_integer(raw.get("etcd_peer_port"), "control.etcd_peer_port", 1),
        nats_port=_integer(raw.get("nats_port"), "control.nats_port", 1),
        startup_timeout=float(raw.get("startup_timeout", 300)),
    )


def _hca_selection(value):
    if value not in HCA_SELECTION:
        raise ConfigError(
            f"fabric.hca_selection must be one of {', '.join(HCA_SELECTION)}"
        )
    return value


def _fabric(raw):
    raw = _mapping(raw, "fabric")
    return Fabric(
        target_utilization=_fraction(
            raw.get("target_utilization"), "invalid fabric.target_utilization"
        ),
        kv_bytes_per_element=float(raw.get("kv_bytes_per_element", 2)),
        recurrent_bytes_per_element=float(raw.get("recurrent_bytes_per_element", 4)),
        conv_bytes_per_element=float(raw.get("conv_bytes_per_element", 2)),
        hca_selection=_hca_selection(raw.get("hca_selection", "all")),
    )


def _worker(raw, allocations):
    raw = _mapping(raw, "worker")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("worker names must be non-empty strings")
    if name in CONTROL_SERVICES:
        raise ConfigError("worker names must not shadow control services")
    if raw.get("role") not in ROLES:
        raise ConfigError(f"{name}.role must be one of {', '.join(ROLES)}")
    tp = _integer(raw.get("tp"), "tp", 1)
    gpu_ids = raw.get("gpu_ids")
    if gpu_ids is not None:
        _sequence(gpu_ids, f"{name}.gpu_ids")
        message = f"{name}.gpu_ids must contain tp unique nonnegative integers"
        if len(gpu_ids) != tp:
            raise ConfigError(message)
        for gpu_id in gpu_ids:
            if not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0:
                raise ConfigError(message)
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ConfigError(message)
        gpu_ids = tuple(gpu_ids)
    allocation_name = raw.get("allocation") or raw.get("node")
    if allocation_name not in allocations:
        raise ConfigError(
            f"{name}: unknown allocation {allocation_name!r}; "
            f"declare it under allocations"
        )
    allocation = allocations[allocation_name]
    node = str(raw.get("node") or allocation.node)
    if node != allocation.node:
        raise ConfigError(
            f"{name}: node {node} does not match allocation "
            f"{allocation_name} on {allocation.node}"
        )
    return Worker(
        name=name,
        role=raw["role"],
        allocation=allocation_name,
        node=node,
        tp=tp,
        system_port=_integer(raw.get("system_port"), f"{name}.system_port", 1),
        nixl_port=_integer(raw.get("nixl_port"), f"{name}.nixl_port", 1),
        event_port=_optional_integer(raw.get("event_port"), f"{name}.event_port", 1),
        gpu_ids=gpu_ids,
        cpus_per_task=_optional_integer(
            raw.get("cpus_per_task"), f"{name}.cpus_per_task", 1
        ),
        numa_node=_optional_integer(raw.get("numa_node"), f"{name}.numa_node", 0),
    )


def _check_ports(control, frontend, workers, control_node):
    """Every listener on a node must have its own port."""
    ports = {control_node: [*control.listeners(), frontend.port]}
    for worker in workers:
        ports.setdefault(worker.node, []).extend(
            port
            for port in (worker.system_port, worker.nixl_port, worker.event_port)
            if port is not None
        )
    for node, values in ports.items():
        if len(values) != len(set(values)) or any(
            not 1 <= port <= 65535 for port in values
        ):
            raise ConfigError(f"invalid or colliding ports on {node}")


def _allocations(raw, recipe):
    """Normalize the two accepted allocation forms into named allocations.

    ``node: 12345`` is shorthand for an allocation named after its node; the
    explicit form is ``name: {node: ..., job: ...}`` for nodes with more than
    one allocation.
    """
    declared = _mapping(raw, f"{recipe}.allocations")
    allocations = {}
    for name, value in declared.items():
        name = str(name)
        if isinstance(value, int) and not isinstance(value, bool):
            node, job = name, value
        elif isinstance(value, Mapping):
            node = str(value.get("node") or name)
            job = value.get("job")
        else:
            raise ConfigError(
                f"{recipe}.allocations.{name}: use a job ID or {{node: ..., job: ...}}"
            )
        if not isinstance(job, int) or isinstance(job, bool) or job <= 0:
            raise ConfigError("allocation IDs must be positive integers")
        if not node:
            raise ConfigError(f"{recipe}.allocations.{name}: node must be set")
        allocations[name] = Allocation(name=name, job=job, node=node)
    jobs = [allocation.job for allocation in allocations.values()]
    if len(jobs) != len(set(jobs)):
        raise ConfigError(f"{recipe}: the same allocation is declared twice")
    return allocations


def _check_gpu_ownership(workers):
    """Workers inside one allocation must claim disjoint GPU ordinals."""
    for name, colocated in _by_allocation(workers).items():
        if len(colocated) < 2:
            continue
        if any(worker.gpu_ids is None for worker in colocated):
            raise ConfigError(
                f"workers sharing allocation {name} must each set gpu_ids "
                "to avoid sharing devices"
            )
        claimed = [gpu_id for worker in colocated for gpu_id in worker.gpu_ids]
        if len(claimed) != len(set(claimed)):
            raise ConfigError(
                f"workers sharing allocation {name} must use disjoint gpu_ids"
            )


def _recipe(name, value, models, control, frontend):
    raw = _mapping(value, f"recipes.{name}")
    control_node = raw.get("control_node")
    allocations = _allocations(raw.get("allocations", {}), name)
    raw_workers = _sequence(raw.get("workers", []), f"{name}.workers")
    nodes = {allocation.node for allocation in allocations.values()}
    if control_node not in nodes or not raw_workers:
        raise ConfigError(
            f"{name}: configure control_node, allocations and workers first"
        )
    control_allocation = raw.get("control_allocation")
    if control_allocation is not None:
        chosen = allocations.get(control_allocation)
        if chosen is None or chosen.node != control_node:
            raise ConfigError(
                f"{name}: control_allocation must name an allocation on {control_node}"
            )
    workers = tuple(_worker(item, allocations) for item in raw_workers)
    if {worker.role for worker in workers} != set(ROLES):
        raise ConfigError("a recipe needs prefill and decode workers")
    if len({worker.name for worker in workers}) != len(workers):
        raise ConfigError("worker names must be unique")
    if len({worker.tp for worker in workers}) != 1:
        raise ConfigError("the pinned hybrid-model connector requires matching P/D TP")
    _check_gpu_ownership(workers)
    _check_ports(control, frontend, workers, control_node)
    if raw.get("model") not in models:
        raise ConfigError(f"unknown model: {raw.get('model')}")
    return Recipe(
        name=name,
        model=raw["model"],
        control_node=control_node,
        allocations=allocations,
        workers=workers,
        control_allocation_name=control_allocation,
    )


def _by_allocation(workers):
    grouped: dict[Any, list[Worker]] = {}
    for worker in workers:
        grouped.setdefault(worker.allocation, []).append(worker)
    return grouped


def load(path=DEFAULT_CONFIG):
    """Read, validate and type the whole file. Raises ConfigError."""
    raw = _read(path)
    models = _models(raw.get("models"))
    control = _control(raw.get("control"))
    frontend = _frontend(raw.get("frontend"))
    return Settings(
        path=Path(path),
        software=_software(raw.get("software")),
        models=models,
        engine=_engine(raw.get("engine")),
        frontend=frontend,
        control=control,
        transport=Transport(
            environment={
                str(key): str(value)
                for key, value in dict(
                    _mapping(raw.get("transport") or {}, "transport").get(
                        "environment", {}
                    )
                ).items()
            }
        ),
        fabric=_fabric(raw.get("fabric")),
        recipes=dict(_mapping(raw.get("recipes"), "recipes")),
    )


def resolve(settings, name="initial-h100", model=None):
    """Select a recipe/model pair, with the remaining cross-checks applied."""
    recipe = settings.recipe(name)
    key = model or recipe.model
    if key not in settings.models:
        raise ConfigError(f"unknown model: {key}")
    return Launch(
        settings=settings,
        recipe=replace(recipe, model=key),
        model=settings.models[key],
    )


def read(path=DEFAULT_CONFIG, name="initial-h100", model=None):
    """Convenience for callers that want a resolved launch straight from a file."""
    return resolve(load(path), name, model)


def recipe_summary(settings):
    """(name, configured?) pairs in file order, for `switchyard config --list`."""
    return [(name, settings.configured(name)) for name in settings.recipes]
