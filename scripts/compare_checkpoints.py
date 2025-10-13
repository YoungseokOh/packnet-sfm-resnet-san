#!/usr/bin/env python3
"""Evaluate multiple depth checkpoints side-by-side and summarise metrics.

This utility automates running the PackNet-SfM evaluation loop for a list of
(checkpoint, config) pairs and prints all returned metrics in a compact table.
It is handy for comparing models that were trained with different input
resolutions (e.g., 640×384 vs. 1920×1536) on the same validation split.

Example
-------
::

    python scripts/compare_checkpoints.py \
        --run hires /path/to/hires.ckpt configs/eval_ncdb_1920_val.yaml \
        --run lowres /path/to/lowres.ckpt configs/eval_ncdb_640_val.yaml

The script will evaluate each checkpoint using the provided override config and
print a metrics table once finished. Use ``--half`` to enable fp16 inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.horovod import hvd_init
from packnet_sfm.utils.logging import pcolor


RunSpec = Tuple[str, str, str]  # (label, checkpoint, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple PackNet-SfM checkpoints on the same split.")
    parser.add_argument(
        "--run",
        metavar=("LABEL", "CHECKPOINT", "CONFIG"),
        nargs=3,
        action="append",
        required=True,
        help=(
            "Triplet specifying the run label, checkpoint (.ckpt) path, and "
            "override config (.yaml). Provide this argument multiple times "
            "to evaluate several checkpoints in a single call."),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable fp16 inference (affects both network weights and inputs).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to store the aggregated metrics as JSON.",
    )
    return parser.parse_args()


def _format_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Flatten and sort metrics dictionary for nicer table output."""
    ordered = {}
    for key in sorted(metrics.keys()):
        val = metrics[key]
        if isinstance(val, (int, float)):
            ordered[key] = float(val)
    return ordered


def evaluate_run(run: RunSpec, use_half: bool) -> Dict[str, float]:
    label, ckpt_path, cfg_path = run
    ckpt_path = Path(ckpt_path)
    cfg_path = Path(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(pcolor(f"\n### Evaluating {label}", 'green'))
    config, state_dict = parse_test_file(str(ckpt_path), str(cfg_path))

    if use_half:
        config.arch.dtype = torch.float16
    else:
        config.arch.dtype = None

    hvd_init()
    trainer = HorovodTrainer(**config.arch)
    model = ModelWrapper(config, load_datasets=True)
    model.load_state_dict(state_dict)

    dtype = trainer.dtype
    if dtype is not None:
        model = model.to('cuda', dtype=dtype)
    else:
        model = model.to('cuda')

    dataloaders = model.test_dataloader()
    metrics = trainer.evaluate(dataloaders, model)
    metrics = _format_metrics(metrics)

    print("Results:")
    for key, value in metrics.items():
        print(f"  {key:>20s}: {value:.6f}")

    return metrics


def main() -> None:
    args = parse_args()
    runs: List[RunSpec] = args.run
    results: Dict[str, Dict[str, float]] = {}

    for run in runs:
        label = run[0]
        metrics = evaluate_run(run, args.half)
        results[label] = metrics

    # Print combined table
    if results:
        print("\n=== Summary ===")
        # Collect union of metric keys
        keys = sorted({k for metrics in results.values() for k in metrics.keys()})
        header = ["metric"] + list(results.keys())
        widths = [max(len(h), 12) for h in header]

        def fmt_row(values: List[str]) -> str:
            return " | ".join(v.ljust(w) for v, w in zip(values, widths))

        print(fmt_row(header))
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))
        for key in keys:
            row = [key]
            for label in results.keys():
                val = results[label].get(key, float('nan'))
                row.append(f"{val:.6f}" if torch.isfinite(torch.tensor(val)) else "nan")
            print(fmt_row(row))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open('w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
