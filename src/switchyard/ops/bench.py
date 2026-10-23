"""Run `vllm bench serve` against a live endpoint, inside the inference runtime.

Two short cases expose prefill and decode behavior without consuming the night.
The measurements are throughput and latency of a synthetic workload; they are
not a task-quality evaluation.
"""

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..paths import ROOT, RUNTIME_RECIPE, expand, python_in


@dataclass(frozen=True)
class Target:
    url: str
    label: str


@dataclass(frozen=True)
class Case:
    name: str
    input_len: int
    output_len: int
    concurrency: int
    prompts: int


CASES = (
    Case("prefill-heavy", 8192, 256, 4, 12),
    Case("decode-heavy", 512, 2048, 8, 24),
)


def case_arguments(launch, case, result_dir, target):
    """The `vllm bench serve` argument vector for one case."""
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        target.url,
        "--endpoint",
        "/v1/completions",
        "--model",
        launch.model.served_name,
        "--tokenizer",
        expand(launch.model.path),
        "--trust-remote-code",
        "--dataset-name",
        "random",
        "--input-len",
        str(case.input_len),
        "--output-len",
        str(case.output_len),
        "--max-concurrency",
        str(case.concurrency),
        "--num-prompts",
        str(case.prompts),
        "--num-warmups",
        "1",
        "--temperature",
        "0",
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        f"{case.name}.json",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metadata",
        f"scenario={case.name}",
        f"target={target.label}",
    ]


def bench_command(launch, case, result_dir, target):
    """Wrap one case in the module/CUDA setup it needs on a compute node."""
    software = launch.settings.software
    python = python_in(software.venv)
    script = "\n".join(
        [
            "set -euo pipefail",
            "cd " + shlex.quote(str(ROOT)),
            "source " + shlex.quote(str(RUNTIME_RECIPE)),
            "switchyard_runtime " + shlex.join([software.cuda_home, *software.modules]),
            "export PATH=" + shlex.quote(str(Path(python).parent)) + ":${PATH}",
            "exec " + shlex.join(case_arguments(launch, case, result_dir, target)),
        ]
    )
    return ["bash", "-lc", script]


def run(launch, base_url, result_dir, target_label, cases=CASES):
    """Execute every case; a failing case stops the sweep."""
    target = Target(url=base_url, label=target_label)
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        command = bench_command(launch, case, result_dir, target)
        subprocess.run(command, check=True)
    return result_dir
