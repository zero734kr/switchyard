"""Install pinned official control-plane binaries without touching the inference venv."""

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import urllib.request
from config import expand, read_config


def main():
    s = read_config()["software"]
    for name, repo, prefix, binary in (
        ("etcd", "etcd-io/etcd", "etcd", "etcd"),
        ("nats", "nats-io/nats-server", "nats-server", "nats-server"),
    ):
        version = s["control_versions"][name]
        destination = Path(expand(s[name]))
        if destination.exists():
            info = subprocess.check_output(
                [str(destination), "--version" if name == "etcd" else "-v"], text=True
            )
            if version.lstrip("v") not in info:
                raise RuntimeError(
                    f"refusing to overwrite {destination}: {info.strip()}"
                )
            print(info.strip())
            continue
        filename = f"{prefix}-{version}-linux-amd64.tar.gz"
        base = f"https://github.com/{repo}/releases/download/{version}/"
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
                m
                for m in archive.getmembers()
                if Path(m.name).name == binary and m.isfile()
            )
            payload = archive.extractfile(member).read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(payload)
        destination.chmod(0o755)
        print(f"installed {destination}; archive sha256={actual}")


if __name__ == "__main__":
    main()
