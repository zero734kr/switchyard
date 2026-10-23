"""Install the pinned official control-plane binaries.

This never touches the inference virtual environment and never upgrades an
existing binary: a mismatched version is an error, not something to overwrite.
"""

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

from ..paths import expand

BINARIES = {
    # name: (GitHub repository, release archive prefix, member name, version flag)
    "etcd": ("etcd-io/etcd", "etcd", "etcd", "--version"),
    "nats": ("nats-io/nats-server", "nats-server", "nats-server", "-v"),
}
ARCHIVE = "{prefix}-{version}-linux-amd64.tar.gz"
BASE = "https://github.com/{repo}/releases/download/{version}/"


def _report_existing(destination, name, version):
    flag = BINARIES[name][3]
    import subprocess

    info = subprocess.check_output([str(destination), flag], text=True)
    if version.lstrip("v") not in info:
        raise RuntimeError(f"refusing to overwrite {destination}: {info.strip()}")
    return info.strip()


def _download(name, version):
    repo, prefix, member_name, _ = BINARIES[name]
    filename = ARCHIVE.format(prefix=prefix, version=version)
    base = BASE.format(repo=repo, version=version)
    sums = urllib.request.urlopen(base + "SHA256SUMS", timeout=60).read().decode()
    expected = next(
        line.split()[0]
        for line in sums.splitlines()
        if line.split()[-1].lstrip("*") == filename
    )
    data = urllib.request.urlopen(base + filename, timeout=120).read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checksum mismatch: {filename}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = next(
            item
            for item in archive.getmembers()
            if Path(item.name).name == member_name and item.isfile()
        )
        payload = archive.extractfile(member).read()
    return payload, actual


def install_control_plane(settings):
    """Install etcd and nats-server at the versions pinned in config.yaml."""
    software = settings.software
    installed = {}
    for name in BINARIES:
        version = software.control_versions[name]
        destination = Path(expand(getattr(software, name)))
        if destination.exists():
            print(_report_existing(destination, name, version))
            installed[name] = str(destination)
            continue
        payload, digest = _download(name, version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(payload)
        destination.chmod(0o755)
        print(f"installed {destination}; archive sha256={digest}")
        installed[name] = str(destination)
    return installed
