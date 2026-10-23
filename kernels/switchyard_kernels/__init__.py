"""Triton kernels and the guarded vLLM adapters that expose them.

``sitecustomize`` is the entry point vLLM picks up through ``PYTHONPATH``; it
calls :func:`switchyard_kernels.install.install` with the model key. Nothing in
this package is imported until that happens, so importing it is always safe on
a machine without torch, triton or vllm.
"""

__all__ = ["install"]
