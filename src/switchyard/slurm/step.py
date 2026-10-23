"""Build the ``srun`` command that runs one service inside an existing allocation.

Each service is started as its own one-task step with ``--overlap`` so several
steps can share one allocation. ``--exact`` is deliberately *not* used for
worker steps: inherited per-task requests would otherwise shrink the step and
the CPU partitioner could not hand out the allocation's full CPU pool.
"""

import shlex
from pathlib import Path

from ..config import CONTROL_SERVICES
from ..paths import ROOT, RUNTIME_RECIPE, expand, python_in

# Inherited per-task requests imply --exact and can shrink a different
# allocation's step. Clear them so placement has the full CPU pool to divide.
INHERITED_TASK_VARIABLES = (
    "SLURM_CPUS_PER_TASK",
    "SRUN_CPUS_PER_TASK",
    "SLURM_CPUS_PER_GPU",
    "SLURM_GPUS_PER_TASK",
    "SLURM_TRES_PER_TASK",
    "SLURM_EXACT",
    "SLURM_EXCLUSIVE",
)

ENTRYPOINT = ROOT / "bin" / "switchyard"


def allocation_of(launch, service):
    """The allocation a service runs in: its worker's, or the control services'."""
    recipe = launch.recipe
    if service in CONTROL_SERVICES:
        return recipe.control_allocation()
    return recipe.allocation_for(recipe.worker(service))


def step(launch, service, run_dir, config_path, model_key, recipe_name="initial-h100"):
    """The full ``env -u ... srun ... bash -lc ...`` command for one service."""
    allocation = allocation_of(launch, service)
    node = allocation.node
    software = launch.settings.software
    setup = [
        "set -euo pipefail",
        "cd " + shlex.quote(str(ROOT)),
        "source " + shlex.quote(str(RUNTIME_RECIPE)),
        "switchyard_runtime " + shlex.join([software.cuda_home, *software.modules]),
        "exec "
        + shlex.join(
            [
                "env",
                "SWITCHYARD_PYTHON=" + interpreter(software),
                str(ENTRYPOINT),
                "role",
                recipe_name,
                "--config",
                str(Path(config_path).resolve()),
                "--model",
                model_key,
                "--service",
                service,
                "--run-dir",
                str(run_dir),
            ]
        ),
    ]
    command = [
        "env",
        *(
            argument
            for variable in INHERITED_TASK_VARIABLES
            for argument in ("-u", variable)
        ),
        "srun",
        f"--jobid={allocation.job}",
        "--cpu-bind=none",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
    ]
    if service in CONTROL_SERVICES:
        # Control services are CPU-only even on a GPU allocation.
        command.append("--gres=none")
    command += ["bash", "-lc", "\n".join(setup)]
    return command


def interpreter(software):
    """The pinned inference interpreter, used by role steps and probes."""
    return python_in(software.venv)


def probe_command(launch, allocation, script, *, exact=True, cpus=1):
    """A small one-off srun step inside ``allocation``, for reachability probes."""
    node = allocation.node
    command = [
        "srun",
        f"--jobid={allocation.job}",
        "--cpu-bind=none",
        "--overlap",
    ]
    if exact:
        command.append("--exact")
    command += ["--nodes=1", "--ntasks=1"]
    if cpus:
        command.append(f"--cpus-per-task={cpus}")
    command += [
        f"--nodelist={node}",
        interpreter(launch.settings.software),
        "-c",
        script,
    ]
    return command


def path_for(value):
    return expand(value)
