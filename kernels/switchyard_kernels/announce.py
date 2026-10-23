"""One place for the JARVIS_* evidence lines the results documents cite.

Every marker is printed once per key and then suppressed: the logs have to stay
greppable without turning into a megabyte of per-token spam.
"""

_announced = set()


def announce(marker, *, key=None, **fields):
    """Print ``MARKER field=value ...`` once for ``key``; return whether it printed."""
    identity = (marker, key if key is not None else tuple(sorted(fields.items())))
    if identity in _announced:
        return False
    _announced.add(identity)
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"{marker} {details}".rstrip(), flush=True)
    return True


def announce_payload(marker, payload, *, key=None):
    """Announce a marker whose evidence is a single pre-rendered payload."""
    identity = (marker, key)
    if identity in _announced:
        return False
    _announced.add(identity)
    print(f"{marker} {payload}", flush=True)
    return True


def reset():
    """Forget what has been announced (tests only)."""
    _announced.clear()
