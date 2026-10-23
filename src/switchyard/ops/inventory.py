"""Print CPU-side topology facts. Starts nothing and initializes no GPU."""

import shutil
import subprocess
from pathlib import Path

#: (heading, argv) pairs; a pair without a heading continues the section above.
COMMANDS = (
    (
        "GPU inventory (metadata only)",
        ("nvidia-smi", "--query-gpu=name,memory.total,pci.bus_id", "--format=csv"),
    ),
    (None, ("nvidia-smi", "topo", "-m")),
    ("Network interfaces", ("ip", "-brief", "address")),
    ("RDMA devices", ("ibstat",)),
    (None, ("ibdev2netdev",)),
)

#: `ucx_info -d` is noisy; these are the lines placement actually reads.
UCX_FILTER = ("Transport:", "Device:", "Type:", "bandwidth:", "memory types:")


def _capture(argv):
    if shutil.which(argv[0]) is None:
        return ""
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def _ucx():
    text = _capture(("ucx_info", "-d"))
    return "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith(UCX_FILTER)
    )


def _rdma_ports():
    lines = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        for name in ("rate", "state"):
            target = port / name
            if target.is_file():
                lines.append(f"{target}: {target.read_text().strip()}")
    return "\n".join(lines)


def inventory(out=print):
    """Everything the old inventory.sh printed, from one callable."""
    out(_capture(("hostname",)))
    for heading, argv in COMMANDS:
        text = _capture(argv)
        if heading:
            out(f"\n{heading}")
        if text:
            out(text)
    ucx = _ucx()
    if ucx:
        out("\nUCX transports")
        out(ucx)
    ports = _rdma_ports()
    if ports:
        out("\nInfiniBand port state and rate")
        out(ports)
