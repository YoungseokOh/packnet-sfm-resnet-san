#!/usr/bin/env python3
"""Render evaluation metrics into matplotlib table images.

This utility is meant for quickly creating shareable PNG tables for reports.

It supports two table types:
- "metrics": abs_rel/sq_rel/rmse/rmse_log/a1/a2/a3
- "summary": depth_type/folder/abs_rel/a1 (or any custom columns)

The script is intentionally dependency-light (matplotlib only).

Example:
  python scripts/visualization/render_metrics_tables.py \
    --out_dir outputs/tables \
    --npu_title "NPU (depth_synthetic)" \
    --npu_metrics '{"abs_rel":0.0932,"sq_rel":0.0764,"rmse":0.4346,"rmse_log":0.1697,"a1":92.66,"a2":96.90,"a3":98.31}' \
    --gpu_title "GPU (depth_synthetic)" \
    --gpu_summary '[{"depth_type":"depth_synthetic","folder":"newest_depth_maps","abs_rel":0.070,"a1":93.6}]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

# Headless-friendly backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_METRICS_ORDER = [
    ("abs_rel", "abs_rel"),
    ("sq_rel", "sq_rel"),
    ("rmse", "rmse"),
    ("rmse_log", "rmse_log"),
    ("a1", "a1 (%)"),
    ("a2", "a2 (%)"),
    ("a3", "a3 (%)"),
]


def render_comparison_table_png(
    title: str,
    subtitle: str | None,
    baseline_label: str,
    baseline: Dict[str, Any],
    other_label: str,
    other: Dict[str, Any],
    out_path: Path,
    order: Sequence[tuple[str, str]] = DEFAULT_METRICS_ORDER,
    figsize=(14.5, 2.8),
) -> None:
    """Render a wide comparison table like:

    Rows: baseline, other, delta (other - baseline)
    Cols: metrics

    Notes:
    - Expects a1/a2/a3 in percent (e.g., 92.66)
    - Delta row uses the same units as inputs.
    """

    keys = [k for k, _ in order]
    col_labels = ["Model"] + [label for _, label in order]

    def fmt(k: str, v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, (int, float)):
            if k in {"a1", "a2", "a3"}:
                return f"{float(v):.4f}"  # match attachment (0.x) OR keep percent? see below
            return f"{float(v):.4f}" if k in {"abs_rel", "sq_rel", "rmse_log"} else f"{float(v):.3f}"
        return str(v)

    # Heuristic: if a1 provided as percent (e.g. 92.66), convert to ratio (0.9266)
    def to_ratio_if_percent(k: str, v: Any) -> Any:
        if v is None or not isinstance(v, (int, float)):
            return v
        if k in {"a1", "a2", "a3"} and v > 1.5:
            return float(v) / 100.0
        return v

    baseline_norm = {k: to_ratio_if_percent(k, baseline.get(k)) for k in keys}
    other_norm = {k: to_ratio_if_percent(k, other.get(k)) for k in keys}
    delta = {
        k: (other_norm.get(k) - baseline_norm.get(k))
        if isinstance(other_norm.get(k), (int, float)) and isinstance(baseline_norm.get(k), (int, float))
        else None
        for k in keys
    }

    rows = []
    rows.append([baseline_label] + [fmt(k, baseline_norm.get(k)) for k in keys])
    rows.append([other_label] + [fmt(k, other_norm.get(k)) for k in keys])
    rows.append([f"Δ ({other_label} - {baseline_label})"] + [fmt(k, delta.get(k)) for k in keys])

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    if subtitle:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)

    header_color = "#2C57A1"  # close to attachment
    baseline_color = "#DDEEFF"
    other_color = "#FFF2CC"
    delta_bad = "#FFCCD5"
    delta_good = "#CCF2D6"

    # Style cells
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_color)
            continue

        # Row backgrounds
        if r == 1:
            cell.set_facecolor(baseline_color)
        elif r == 2:
            cell.set_facecolor(other_color)
        elif r == 3:
            # Delta row: green if improvement, red if worse.
            if c == 0:
                cell.set_facecolor(delta_bad)
                cell.set_text_props(weight="bold")
            else:
                key = keys[c - 1]
                dv = delta.get(key)
                # Lower is better for errors, higher is better for accuracies
                higher_better = key in {"a1", "a2", "a3"}
                improved = (dv > 0) if higher_better else (dv < 0)
                cell.set_facecolor(delta_good if improved else delta_bad)

    # Bold first column
    for rr in range(1, 4):
        table[(rr, 0)].set_text_props(weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if key in {"a1", "a2", "a3"}:
            return f"{float(value):.2f}"  # already in percent
        return f"{float(value):.4f}"
    return str(value)


def render_kv_table_png(
    title: str,
    metrics: Dict[str, Any],
    out_path: Path,
    order: Sequence[tuple[str, str]] = DEFAULT_METRICS_ORDER,
    figsize=(6.5, 3.0),
) -> None:
    """Render a simple 2-column table: Metric | Value."""
    keys = [k for k, _ in order if k in metrics]
    rows = [[label, _format_value(k, metrics.get(k))] for k, label in order if k in metrics]

    # Add any extra keys not in default order
    extra = [k for k in metrics.keys() if k not in set(keys)]
    for k in sorted(extra):
        rows.append([k, _format_value(k, metrics.get(k))])

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 1.35)

    # Styling header row
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2C3E50")
        else:
            cell.set_facecolor("#ECF0F1" if r % 2 == 0 else "#FFFFFF")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_dataframe_table_png(
    title: str,
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    out_path: Path,
    figsize=(8.5, 2.6),
) -> None:
    """Render an N-column table from list-of-dicts."""
    cell_text: List[List[str]] = []
    for row in rows:
        cell_text.append([_format_value(col, row.get(col)) for col in columns])

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=list(columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 1.35)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2C3E50")
        else:
            cell.set_facecolor("#ECF0F1" if r % 2 == 0 else "#FFFFFF")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render NPU/GPU metrics tables to PNG")
    p.add_argument("--out_dir", type=str, default="outputs/tables", help="Output directory for PNGs")

    # NPU (metrics table)
    p.add_argument("--npu_title", type=str, default="NPU Results", help="Title for NPU table")
    p.add_argument(
        "--npu_metrics",
        type=str,
        default=None,
        help="JSON dict with keys abs_rel/sq_rel/rmse/rmse_log/a1/a2/a3 (a1-3 should be in percent)",
    )

    # GPU (summary table)
    p.add_argument("--gpu_title", type=str, default="GPU Results", help="Title for GPU table")
    p.add_argument(
        "--gpu_summary",
        type=str,
        default=None,
        help="JSON list of rows, e.g. [{\"depth_type\":...,\"folder\":...,\"abs_rel\":0.070,\"a1\":93.6}]",
    )

    # Horizontal comparison table (like the attachment)
    p.add_argument("--comparison_title", type=str, default=None, help="Title for horizontal comparison table")
    p.add_argument("--comparison_subtitle", type=str, default=None, help="Subtitle line under title")
    p.add_argument("--baseline_label", type=str, default="PyTorch FP32", help="Row label for baseline")
    p.add_argument("--baseline_metrics", type=str, default=None, help="JSON dict for baseline metrics")
    p.add_argument("--other_label", type=str, default="NPU", help="Row label for other")
    p.add_argument("--other_metrics", type=str, default=None, help="JSON dict for other metrics")
    p.add_argument("--comparison_out", type=str, default=None, help="Filename for comparison PNG")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.npu_metrics:
        npu_metrics = json.loads(args.npu_metrics)
        render_kv_table_png(
            title=args.npu_title,
            metrics=npu_metrics,
            out_path=out_dir / "npu_metrics.png",
        )
        print(f"Saved: {out_dir / 'npu_metrics.png'}")

    if args.gpu_summary:
        gpu_rows = json.loads(args.gpu_summary)
        # Default columns for your use-case
        cols = ["depth_type", "folder", "abs_rel", "a1"]
        render_dataframe_table_png(
            title=args.gpu_title,
            rows=gpu_rows,
            columns=cols,
            out_path=out_dir / "gpu_summary.png",
            figsize=(8.8, 2.4 + 0.45 * max(1, len(gpu_rows))),
        )
        print(f"Saved: {out_dir / 'gpu_summary.png'}")

    if args.baseline_metrics and args.other_metrics:
        baseline = json.loads(args.baseline_metrics)
        other = json.loads(args.other_metrics)
        out_name = args.comparison_out or "fp32_vs_npu_comparison.png"
        render_comparison_table_png(
            title=args.comparison_title or "Dual-Head Model: PyTorch FP32 vs NPU Comparison",
            subtitle=args.comparison_subtitle,
            baseline_label=args.baseline_label,
            baseline=baseline,
            other_label=args.other_label,
            other=other,
            out_path=out_dir / out_name,
        )
        print(f"Saved: {out_dir / out_name}")

    if not args.npu_metrics and not args.gpu_summary and not (args.baseline_metrics and args.other_metrics):
        print("Nothing to render. Provide --npu_metrics/--gpu_summary and/or --baseline_metrics/--other_metrics JSON.")


if __name__ == "__main__":
    main()
