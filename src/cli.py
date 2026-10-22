"""Shared command-line parsing and run-directory handling for entry points."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import distributed

RESOLVED_CONFIG_FILE = "config.yaml"
GIT_COMMIT_FILE = "git_commit.txt"


@dataclass(frozen=True)
class RunArguments:
    """Parsed entry-point arguments."""

    config_path: Path
    output_dir: Path
    overrides: tuple[str, ...]


def parse_run_arguments(description: str) -> RunArguments:
    """Parse ``--config``, ``--output-dir``, and repeated ``--set key=value`` overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run's outputs")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override, e.g. pairing.alpha=0.5")
    namespace = parser.parse_args()
    return RunArguments(
        config_path=namespace.config, output_dir=namespace.output_dir, overrides=tuple(namespace.overrides)
    )


@dataclass(frozen=True)
class DistributedContext:
    """Placement of this process; ``world_size == 1`` for a plain single-process run."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        """Whether this process writes outputs and logs."""
        return self.rank == 0


def start_distributed() -> DistributedContext:
    """Join the process group when launched by ``torchrun`` with ``WORLD_SIZE > 1`` (NCCL on CUDA, else gloo)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext(rank=0, local_rank=0, world_size=1)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return DistributedContext(rank=distributed.get_rank(), local_rank=local_rank, world_size=world_size)


def select_device(*, configured_device: str, context: DistributedContext) -> torch.device:
    """Return the configured device, or this process's own GPU in a multi-GPU CUDA run."""
    device = torch.device(configured_device)
    if context.world_size > 1 and device.type == "cuda":
        return torch.device("cuda", context.local_rank)
    return device


def wait_for_all_processes(context: DistributedContext) -> None:
    """Block until every process reaches this point (no-op for a single process)."""
    if context.world_size > 1:
        distributed.barrier()


def stop_distributed(context: DistributedContext) -> None:
    """Leave the process group (no-op for a single process)."""
    if context.world_size > 1:
        distributed.destroy_process_group()


def make_run_directory(output_dir: Path) -> None:
    """Create a fresh run directory and record the git commit.

    Raises:
        FileExistsError: If the directory already exists, so earlier results are never overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout
    (output_dir / GIT_COMMIT_FILE).write_text(f"{commit or 'unknown'}{' (dirty)' if status.strip() else ''}\n")


def seed_global_generators(seed: int) -> None:
    """Seed Python, NumPy, and torch global RNGs (model initialization, augmentations)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def append_json_line(path: Path, record: dict[str, object]) -> None:
    """Append one JSON record to a JSON-lines file."""
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def write_json(path: Path, record: dict[str, object]) -> None:
    """Write a JSON file, refusing to overwrite an existing one."""
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
