"""GPU-to-HCA proximity, parsed from ``nvidia-smi topo -m``.

The sysfs ``numa_node`` of an InfiniBand device is not a reliable guide: on
g212, ``mlx5_1`` reports NUMA node 3 on PCI domain 0001 while serving GPUs on
domain 0000 at NUMA 0/1. The topo matrix walks the actual PCI hierarchy and is
the only trustworthy local signal, so proximity is taken from it.

Grades, best first: PIX, PXB, PHB, NODE, SYS. SYS means the path crosses the
CPU interconnect, i.e. there is no locally attached HCA.
"""

import re
from dataclasses import dataclass

GRADES = ("PIX", "PXB", "PHB", "NODE", "SYS")
RANK = {grade: index for index, grade in enumerate(GRADES)}
#: Anything at this grade or worse is not worth pinning to.
CROSS_SOCKET = "SYS"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GPU = re.compile(r"^GPU\d+$")
NIC = re.compile(r"^NIC\d+$")
NIC_LEGEND = re.compile(r"^(NIC\d+):\s+(\S+)$")


@dataclass(frozen=True)
class Matrix:
    """The GPU/NIC block of one ``nvidia-smi topo -m`` table."""

    gpus: tuple[str, ...]
    nics: tuple[str, ...]
    rows: dict[str, dict[str, str]]

    def grade(self, gpu, nic):
        return self.rows.get(gpu, {}).get(nic)

    def best_nics(self, gpu):
        """Devices tied for the best grade from ``gpu``, plus that grade."""
        graded = {
            nic: RANK[value]
            for nic, value in self.rows.get(gpu, {}).items()
            if value in RANK
        }
        if not graded:
            return (), None
        best = min(graded.values())
        chosen = tuple(sorted(nic for nic, rank in graded.items() if rank == best))
        return chosen, GRADES[best]


def parse(text):
    """Parse the matrix; returns None when the table is absent or unreadable."""
    lines = [ANSI.sub("", line) for line in text.splitlines()]
    legend = {}
    for line in lines:
        match = NIC_LEGEND.match(line.strip())
        if match:
            legend[match.group(1)] = match.group(2)

    header = next((line for line in lines if line.lstrip().startswith("GPU0")), None)
    if header is None:
        return None
    columns = [column.strip() for column in header.split("\t") if column.strip()]
    gpus = tuple(column for column in columns if GPU.match(column))
    nics = tuple(column for column in columns if NIC.match(column))

    rows = {}
    for line in lines:
        fields = [field.strip() for field in line.split("\t")]
        if not fields or not fields[0].startswith("GPU"):
            continue
        gpu = fields[0]
        # Keep positional alignment: CPU/NUMA affinity cells are often empty,
        # and dropping them would shift the NIC grades left.
        values = fields[1:][: len(columns) - 1]
        if len(values) < len(columns) - 1:
            continue
        rows[gpu] = dict(zip((*gpus, *nics), values, strict=False))
    if not rows:
        return None
    named = {column: legend[column] for column in nics if column in legend}
    return Matrix(
        gpus=gpus,
        nics=tuple(named.values()),
        rows={
            gpu: {
                named[column]: value for column, value in row.items() if column in named
            }
            for gpu, row in rows.items()
        },
    )


def best_hcas(matrix, ordinals):
    """Best HCA devices across every GPU ordinal a worker owns."""
    if matrix is None:
        return (), None
    chosen, best = [], None
    for ordinal in ordinals:
        name = f"GPU{ordinal}"
        devices, grade = matrix.best_nics(name)
        if not devices:
            continue
        if best is None or RANK[grade] < RANK[best]:
            chosen, best = list(devices), grade
        elif RANK[grade] == RANK[best]:
            chosen.extend(device for device in devices if device not in chosen)
    if best is None:
        return (), None
    return tuple(sorted(chosen)), best


def pinnable(grade):
    """Only pin when the path stays inside a NUMA node."""
    return grade is not None and RANK[grade] < RANK[CROSS_SOCKET]
