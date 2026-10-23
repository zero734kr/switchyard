"""Repository anchors and the handful of path expansions used everywhere."""

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
# src/switchyard -> src -> repository root. SWITCHYARD_ROOT exists so an
# installed copy can still be pointed at a checkout's config and recipes.
ROOT = Path(os.environ.get("SWITCHYARD_ROOT") or PACKAGE.parents[1]).resolve()
DEFAULT_CONFIG = ROOT / "config.yaml"
KERNELS = ROOT / "kernels"
RUNTIME_RECIPE = ROOT / "recipes" / "jarvis.sh"
RUNS = ROOT / "runs"


def expand(value):
    """Expand ~ and $VARS in a configuration path."""
    return os.path.expanduser(os.path.expandvars(str(value)))


def python_in(venv):
    """Interpreter of an existing virtual environment, without activating it."""
    return str(Path(expand(venv)) / "bin" / "python")
