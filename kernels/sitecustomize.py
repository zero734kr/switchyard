"""Opt-in process-local adapters inherited by vLLM worker subprocesses.

vLLM imports whatever ``sitecustomize`` it finds on ``PYTHONPATH``; the launcher
only points ``PYTHONPATH`` at this directory when ``engine.kernels: custom``.
Adapters are therefore opt-in, per-process, and never touch installed packages.
"""

import os
import sys

model = os.environ.get("SWITCHYARD_CUSTOM_KERNELS")
if model:
    try:
        from switchyard_kernels.install import install

        install(model)
    except Exception:
        import traceback

        traceback.print_exc()
        # Python normally swallows sitecustomize exceptions; fail visibly instead.
        sys.stderr.flush()
        os._exit(1)
