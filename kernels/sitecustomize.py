"""Opt-in process-local adapters inherited by vLLM worker subprocesses."""

import os
import sys

model = os.environ.get("SWITCHYARD_CUSTOM_KERNELS")
if model:
    try:
        for variable in list(os.environ):
            if variable.startswith("JARVIS_"):
                os.environ.pop(variable)
        from importlib.metadata import version

        if version("vllm") != "0.28.0":
            raise RuntimeError("kernel adapters require the validated vLLM 0.28.0 API")
        # Internal flags are adapter details, not user-facing recipe names.
        os.environ["JARVIS_PROJECTION_PAIR"] = "1"
        os.environ["JARVIS_GDN_TILED"] = "public"
        if model == "dense":
            from jarvis_dense import install

            install()
        elif model == "moe":
            from original_moe import install

            install()
        else:
            raise ValueError(f"unknown custom kernel model: {model}")
        from recipe_plugin import install

        install()
        print(f"SWITCHYARD_KERNEL_ADAPTER_LOADED model={model}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        # Python normally swallows sitecustomize exceptions; fail visibly instead.
        sys.stderr.flush()
        os._exit(1)
