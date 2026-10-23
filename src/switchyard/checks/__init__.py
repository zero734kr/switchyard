"""One-shot, independent checks against a running deployment."""

from .smoke import check_api

__all__ = ["check_api"]
