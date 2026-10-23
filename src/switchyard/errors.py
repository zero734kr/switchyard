"""Exceptions that already carry a message worth showing a user."""


class SwitchyardError(Exception):
    """Base class; the CLI prints these without a traceback."""


class ConfigError(SwitchyardError, ValueError):
    """config.yaml is malformed or describes an unusable deployment."""


class PlacementError(SwitchyardError, RuntimeError):
    """The requested CPU, memory or GPU placement cannot be honored."""


class AllocationError(SwitchyardError, RuntimeError):
    """An existing Slurm allocation is missing, expired or owned by someone else."""


class LaunchError(SwitchyardError, RuntimeError):
    """A service exited or never became ready."""
