"""
Policy Training Entry Point
---------------------------
Placeholder CLI for training a robot policy from recorded episodes.

This file intentionally avoids importing PyTorch/JAX/etc. until a framework is
chosen. It documents the training stages and gives the repo a stable command
surface for future work.

Expected future work:
- Load EpisodeLogger datasets from POLICY_DATASET_ROOT.
- Validate camera/state/action alignment.
- Build train/validation splits by episode, not by individual timestep.
- Train the selected policy model.
- Save checkpoints and normalization statistics together.
- Write evaluation metrics and rollout videos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import config as cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a robot policy from logged episodes.")
    parser.add_argument(
        "--dataset-root",
        default=getattr(cfg, "POLICY_DATASET_ROOT", "data/policy"),
        help="Directory containing episode folders.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default=getattr(cfg, "POLICY_CHECKPOINT_PATH", "checkpoints/policy/latest"),
        help="Where the trained policy checkpoint should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the planned training stages without training.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root)
    checkpoint_out = Path(args.checkpoint_out)

    print("Policy training is scaffolded but not implemented yet.")
    print(f"Dataset root: {dataset_root}")
    print(f"Checkpoint output: {checkpoint_out}")
    print("Next implementation steps:")
    print("1. Validate episode metadata and steps.jsonl files.")
    print("2. Load image/state/action tensors with synchronized timestamps.")
    print("3. Choose the model architecture and action normalization.")
    print("4. Train, evaluate, and save checkpoint plus normalization stats.")

    if not args.dry_run:
        print("No training was run because the model and dataset loader are placeholders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
