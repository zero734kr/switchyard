"""Install the adapters selected by ``SWITCHYARD_CUSTOM_KERNELS``."""

from .announce import announce

VLLM_VERSION = "0.28.0"


def _adapter(model):
    if model == "dense":
        from .adapters import dense

        return dense
    if model == "moe":
        from .adapters import moe

        return moe
    raise ValueError(f"unknown custom kernel model: {model}")


def install(model):
    """Patch the guarded operators for ``model`` and the shared GDN paths."""
    from importlib.metadata import version

    installed = version("vllm")
    if installed != VLLM_VERSION:
        raise RuntimeError(
            f"kernel adapters require the validated vLLM {VLLM_VERSION} API, "
            f"found {installed}"
        )
    _adapter(model).install()
    from .adapters import gdn

    gdn.install()
    announce("SWITCHYARD_KERNEL_ADAPTER_LOADED", key=model, model=model)
