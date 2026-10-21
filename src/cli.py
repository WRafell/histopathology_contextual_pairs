"""Shared command-line parsing and run-directory handling for entry points."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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
