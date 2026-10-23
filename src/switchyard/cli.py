"""The ``switchyard`` command line: one entry point for every operation.

The subcommands fall into three groups:

* act on an existing allocation: ``plan``, ``preflight``, ``control-check``, ``run``
* inspect the deployment: ``config``, ``topology``, ``monitor``, ``smoke``,
  ``inventory``
* measure or set up: ``fabric``, ``bench``, ``bootstrap``

``role`` is internal: ``run`` re-execs into it on the target node through srun.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from . import __version__, analysis, checks, config, launcher, ops, paths
from .errors import SwitchyardError


def _add_common(parser):
    parser.add_argument(
        "recipe",
        nargs="?",
        default="initial-h100",
        help="recipe name from config.yaml (default: initial-h100)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=paths.DEFAULT_CONFIG,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model",
        choices=("dense", "moe"),
        help="override the recipe's model",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="switchyard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("plan", "print the srun commands a run would execute"),
        ("preflight", "check allocations, runtime versions, model and binaries"),
        ("control-check", "start the control plane and verify it from every node"),
        ("run", "start all services and serve until interrupted"),
    ):
        _add_common(commands.add_parser(name, help=help_text))

    role = commands.add_parser("role", help="internal: start one service")
    _add_common(role)
    role.add_argument("--service", required=True)
    role.add_argument("--run-dir", type=Path, required=True)

    settings = commands.add_parser("config", help="print or validate configuration")
    _add_common(settings)
    settings.add_argument("--check", action="store_true", help="validate and exit")
    settings.add_argument("--list", action="store_true", help="list recipes and exit")
    settings.add_argument(
        "--json", action="store_true", help="emit the resolved values as JSON"
    )

    fabric = commands.add_parser(
        "fabric", help="model the prefill-to-decode transfer budget"
    )
    _add_common(fabric)
    fabric.add_argument("--model-config", type=Path)
    fabric.add_argument("--prompt-tokens", type=int, default=8192)
    rates = fabric.add_mutually_exclusive_group(required=True)
    rates.add_argument("--requests-per-second", type=float)
    rates.add_argument(
        "--prefill-tokens-per-second",
        type=float,
        help="aggregate uncached token throughput",
    )
    fabric.add_argument("--cache-hit-rate", type=float, default=0)
    fabric.add_argument(
        "--decode-prefix-reuse",
        action="store_true",
        help="only use after connector reuse is verified",
    )
    fabric.add_argument("--state-copies", type=int, default=1)
    fabric.add_argument(
        "--transferred-kv-tokens",
        type=int,
        help="override with measured/aligned transfer count",
    )
    fabric.add_argument(
        "--effective-gbps",
        type=float,
        help="measured application payload bandwidth, not NIC nameplate",
    )

    monitor = commands.add_parser(
        "monitor", help="sample live NIXL bytes and RDMA port counters"
    )
    _add_common(monitor)
    monitor.add_argument(
        "--window", type=float, default=10.0, help="measurement window in seconds"
    )
    monitor.add_argument(
        "--fabric-gbps", type=float, help="known shared path capacity in Gbps"
    )
    monitor.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )

    smoke = commands.add_parser("smoke", help="check the serving API end to end")
    _add_common(smoke)
    smoke.add_argument("--base-url", help="defaults to the configured frontend")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--concurrency", type=int, default=8)

    bench = commands.add_parser("bench", help="run the two vllm bench serve cases")
    _add_common(bench)
    bench.add_argument("--base-url", help="defaults to the configured frontend")
    bench.add_argument("--result-dir", type=Path, required=True)
    bench.add_argument("--target", default="cross-node")

    layout = commands.add_parser(
        "topology",
        help="report GPU/NUMA/HCA layout and the fabric choice per worker",
    )
    _add_common(layout)
    layout.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )

    commands.add_parser("inventory", help="print GPU, network and RDMA topology")
    bootstrap = commands.add_parser(
        "bootstrap", help="install the pinned etcd and nats-server binaries"
    )
    bootstrap.add_argument(
        "--config",
        type=Path,
        default=paths.DEFAULT_CONFIG,
        help="path to config.yaml",
    )
    return parser


def _launch(args):
    settings = config.load(args.config)
    return config.resolve(settings, args.recipe, args.model)


def _frontend_url(launch):
    return f"http://{launch.recipe.control_node}:{launch.settings.frontend.port}"


def _config(args):
    settings = config.load(args.config)
    if args.list:
        for name, configured in config.recipe_summary(settings):
            print(name, "configured" if configured else "awaiting allocations")
        return 0
    launch = config.resolve(settings, args.recipe, args.model)
    if args.check:
        engine = settings.engine
        print(
            f"{args.recipe}: valid; speculation={engine.speculative_method} "
            f"({engine.speculative_tokens} tokens), "
            f"prefix cache={engine.prefix_caching}, kernels={engine.kernels}"
        )
        return 0
    values = _config_values(launch)
    if args.json:
        print(json.dumps(values, indent=2))
        return 0
    for key, value in values.items():
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


def _config_values(launch):
    """The names shell wrappers used to depend on, kept for ad-hoc scripting."""
    settings = launch.settings
    return {
        "PROFILE": launch.recipe.name,
        "EXPECTED_NODE": launch.recipe.control_node,
        "DYNAMO_VENV": paths.expand(settings.software.venv),
        "MODEL_PATH": paths.expand(launch.model.path),
        "MODEL_NAME": launch.model.served_name,
        "CUDA_HOME": settings.software.cuda_home,
        "GCC_MODULE": settings.software.modules[0],
        "CUDA_MODULE": settings.software.modules[1],
        "MAX_NUM_BATCHED_TOKENS": str(settings.engine.max_num_batched_tokens),
    }


def _fabric(args):
    launch = _launch(args)
    settings = launch.settings
    if args.prompt_tokens <= 0 or not 0 <= args.cache_hit_rate < 1:
        raise SwitchyardError("invalid prompt length or cache-hit rate")
    if args.state_copies < 1:
        raise SwitchyardError("state copies must be at least 1")
    if args.transferred_kv_tokens is not None and args.transferred_kv_tokens < 0:
        raise SwitchyardError("transferred KV tokens must be nonnegative")
    if args.requests_per_second is not None:
        rps = args.requests_per_second
    else:
        rps = args.prefill_tokens_per_second / (
            args.prompt_tokens * (1 - args.cache_hit_rate)
        )
    if rps <= 0 or (args.effective_gbps is not None and args.effective_gbps <= 0):
        raise SwitchyardError("rates must be positive")
    source = args.model_config or analysis.budget.checkpoint_path(launch)
    budget = analysis.budget.estimate(
        settings,
        analysis.budget.model_text_config(source),
        args.prompt_tokens,
        rps,
        args.cache_hit_rate,
        args.decode_prefix_reuse,
        args.state_copies,
        args.transferred_kv_tokens,
    )
    result = (
        analysis.budget.measure(budget, args.effective_gbps)
        if args.effective_gbps
        else budget.as_dict()
    )
    print(json.dumps(result, indent=2))
    return 0


def _monitor(args):
    if args.window <= 0:
        raise SwitchyardError("--window must be positive")
    if args.fabric_gbps is not None and args.fabric_gbps <= 0:
        raise SwitchyardError("--fabric-gbps must be positive")
    launch = _launch(args)
    result = analysis.monitor.collect(launch, args.window, args.fabric_gbps)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        analysis.monitor.report(result)
    return 0


def _smoke(args):
    launch = _launch(args)
    result = checks.smoke.run(
        args.base_url or _frontend_url(launch),
        launch.model.served_name,
        args.output,
        args.concurrency,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


def _bench(args):
    launch = _launch(args)
    ops.bench.run(
        launch,
        args.base_url or _frontend_url(launch),
        args.result_dir,
        args.target,
    )
    return 0


def _run(args):
    launch = _launch(args)
    if args.command == "role":
        launcher.role(launch, args.service, args.run_dir, args.config)
        return 0
    run_dir = getattr(args, "run_dir", None) or _next_run_directory()
    if args.command == "plan":
        launcher.plan(launch, run_dir, args.config)
        return 0
    if args.command == "preflight":
        launcher.preflight(launch)
        return 0
    launcher.run(launch, run_dir, check_only=args.command == "control-check")
    return 0


def _topology(args):
    launch = _launch(args)
    try:
        views = ops.topology.inspect(launch)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SwitchyardError(f"topology probe failed: {error}") from error
    data = ops.topology.report(launch, views, verbose=not args.json)
    if args.json:
        print(json.dumps(data, indent=2))
    return 0


def _bootstrap(args):
    ops.bootstrap.install_control_plane(config.load(args.config))
    return 0


def _inventory(_args):
    ops.inventory()
    return 0


def _next_run_directory():
    from .services.supervisor import run_directory

    return run_directory()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "config": _config,
        "bootstrap": _bootstrap,
        "topology": _topology,
        "fabric": _fabric,
        "monitor": _monitor,
        "smoke": _smoke,
        "bench": _bench,
        "inventory": _inventory,
        "plan": _run,
        "preflight": _run,
        "control-check": _run,
        "run": _run,
        "role": _run,
    }
    try:
        return handlers[args.command](args) or 0
    except KeyboardInterrupt:
        print("interrupted; allocations were not cancelled", file=sys.stderr)
        return 130
    except SwitchyardError as error:
        print(f"switchyard: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
