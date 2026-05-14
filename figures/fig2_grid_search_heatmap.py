#!/usr/bin/env python3
"""plot.py
Generate side‑by‑side heatmaps of
  • highest validation accuracy achieved (left)
  • overall compression metric (right)
for every (batch_compression, token_compression) configuration contained in a
results directory produced by your split‑learning experiments.

Usage
-----
$ python plot.py --results_dir /path/to/results \
                --output /path/to/heatmaps.png

Dependencies: Python ≥3.8, pandas, numpy, matplotlib.
Install with: pip install pandas matplotlib numpy
"""

from __future__ import annotations
import argparse
import json
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
from matplotlib import rcParams

font_size = 22
params = {
    'legend.fontsize': 13,
   'axes.labelsize': font_size,
   'axes.linewidth': 1,
   'font.size': font_size,
   # 'legend.fontsize': font_size-2,
   'xtick.labelsize': font_size,
   'xtick.major.size': 2,
   'ytick.labelsize': font_size,
   'ytick.major.size': 2,
   'text.usetex': False,
   # 'figure.figsize': [4*0.9,3*0.9],
   'legend.loc': 'lower right'


}
rcParams.update(params)

PARAM_DIR_RE = re.compile(
    r"params=\{.*'token_compression'\s*_\s*(0\.\d+).*'batch_compression'\s*_\s*(0\.\d+).*}",
    re.DOTALL
)

def extract_metrics(directory: str) -> Tuple[float, float, float, float]:
    """Return (batch_c, token_c, max_val_acc, compression) for *directory*.

    Raises ValueError if the directory name or JSON file is malformed.
    """
    match = PARAM_DIR_RE.fullmatch(os.path.basename(directory))
    if not match:
        raise ValueError("Directory name does not match expected pattern: " + directory)

    token_c = float(match.group(1))
    batch_c = float(match.group(2))

    json_path = os.path.join(directory, "training_results.json")
    with open(json_path, "r") as f:
        results = json.load(f)

    val_accuracies = results.get("Val accuracies", [])
    if not val_accuracies:
        raise ValueError(f"No validation accuracies in {json_path}")
    max_val_acc = val_accuracies[-1]

    compression = results.get("Compression")
    if compression is None:
        # Try alternative keys, if any
        compression = results.get("Compression ratio")
        if compression is None:
            raise ValueError(f"No compression metric in {json_path}")

    return batch_c, token_c, max_val_acc, compression


def gather_dataframe(results_dir: str) -> pd.DataFrame:
    """Scan *results_dir* and build a DataFrame with metrics for each run."""
    data = []
    for entry in os.scandir(results_dir):
        if not entry.is_dir():
            continue
        try:
            metrics = extract_metrics(entry.path)
            data.append(metrics)
        except (ValueError, FileNotFoundError) as exc:
            # Skip malformed folders but notify in console.
            print(f"[warn] {exc}")

    if not data:
        raise RuntimeError("No valid experiment folders found in " + results_dir)

    return pd.DataFrame(
        data,
        columns=["batch_compression", "token_compression", "val_acc", "compression"],
    )


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a pivoted DataFrame indexed by batch_compression (rows) and
    token_compression (cols) for *metric* values."""
    pivot = (
        df.pivot(index="token_compression", columns="batch_compression", values=metric)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return pivot
def make_heatmaps(df: pd.DataFrame, output_file_prefix: str) -> None:
    """Create and save two heatmap figures separately."""

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    acc_grid = pivot_metric(df, "val_acc")
    comp_grid = pivot_metric(df, "compression")

    token_vals = np.array(acc_grid.columns.values, dtype=float)  # k/n (x-axis)
    batch_vals = np.array(acc_grid.index.values, dtype=float)    # T/B (y-axis)


    token_vals_dense = np.linspace(token_vals.min(), token_vals.max(), 500)
    batch_line_dense = token_vals_dense 

    valid = (batch_line_dense >= batch_vals.min()) & (batch_line_dense <= batch_vals.max())
    token_vals_dense = token_vals_dense[valid]
    batch_line_dense = batch_line_dense[valid]

    x_guide = np.interp(token_vals_dense, token_vals, np.arange(len(token_vals)))  # x = k/n
    y_guide = np.interp(batch_line_dense, batch_vals, np.arange(len(batch_vals)))  # y = T/B

    # === First heatmap: Validation Accuracy ===
    fig_acc, ax_acc = plt.subplots(figsize=(7, 7.5), constrained_layout=True)
    cmap_acc = plt.get_cmap("viridis")
    im0 = ax_acc.imshow(acc_grid, origin="lower", aspect="auto", cmap=cmap_acc, norm=PowerNorm(gamma=2.5))

    ax_acc.set_ylabel("k/n")
    ax_acc.set_xlabel("T/B")
    
    # Corrected tick assignment: Y-axis corresponds to Rows (Index), X-axis to Columns
    ax_acc.set_yticks(range(len(acc_grid.index)))
    ax_acc.set_yticklabels(acc_grid.index)
    ax_acc.set_xticks(range(len(acc_grid.columns)))
    ax_acc.set_xticklabels(acc_grid.columns)

    cbar0 = fig_acc.colorbar(im0, ax=ax_acc, pad=0.02, shrink=0.9, orientation="horizontal")

    # --- Overlay Iso-Compression Contours ---
    # Calculate compression grid Z = Token (Y) * Batch (X)
    # Using float values from the index/columns
    y_vals = np.array(acc_grid.index, dtype=float)
    x_vals = np.array(acc_grid.columns, dtype=float)
    Z = np.outer(y_vals, x_vals)  # shape matches acc_grid (Rows x Cols)

    # Choose contour levels (typical compression ratios)
    # You can adjust these levels as desired
    levels = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # Filter to only levels that actually exist in the Z range
    levels = [lvl for lvl in levels if Z.min() <= lvl <= Z.max()]

    # Draw contours
    # Since imshow uses index coordinates 0..N, and Z matches that shape, 
    # contour(Z) works directly in index space.
    cntr = ax_acc.contour(Z, levels=levels, colors='white', linestyles='dashed', linewidths=2.5)

    # Add legend for iso-compression lines
    legend_line = Line2D([0], [0], color='white', lw=2.5, linestyle='dashed')
    ax_acc.legend([legend_line], ["Same ξ lines"], loc="lower left", facecolor="black",fontsize="25", framealpha=0.4, labelcolor="white")
    
    # Label the contours
    # ax_acc.clabel(cntr, inline=True, fontsize=12, fmt='%g', colors='white', use_clabeltext=True)
    # ----------------------------------------


    # ax_acc.plot(x_coords, y_coords, color="blue", linestyle="--", linewidth=3, label="Nuova")
    # ax_acc.plot(x_guide, y_guide, color="red", linestyle="--", linewidth=3, label=r"$k/n = T/B$")
    # ax_acc.legend(loc='lower right')

    acc_file = f"{output_file_prefix}_accuracy.pdf"
    fig_acc.savefig(acc_file, dpi=300, format="pdf")
    fig_acc.savefig(f"{output_file_prefix}_accuracy.jpeg", dpi=300)
    print(f"[info] Accuracy heatmap saved to {acc_file}")

    # === Second heatmap: Compression ===
    fig_comp, ax_comp = plt.subplots(figsize=(7, 7.5), constrained_layout=True)
    cmap_comp = plt.get_cmap("plasma")
    im1 = ax_comp.imshow(comp_grid, origin="lower", aspect="auto", cmap=cmap_comp)

    ax_comp.set_ylabel("k/n")
    ax_comp.set_xlabel("T/B")
    ax_comp.set_yticks(range(len(comp_grid.columns)))
    ax_comp.set_yticklabels(comp_grid.columns)
    ax_comp.set_xticks(range(len(comp_grid.index)))
    ax_comp.set_xticklabels(comp_grid.index)

    cbar1 = fig_comp.colorbar(im1, ax=ax_comp, pad=0.02, shrink=0.9, orientation="horizontal")

    # ax_comp.plot(x_coords, y_coords, color="red", linestyle="--", linewidth=3, label="Custom boundary")
    # ax_comp.plot(x_guide, y_guide, color="blue", linestyle=":", linewidth=2, label=r"$k/n = (T/B)^4$")
    # ax_comp.legend(loc='lower right')

    comp_file = f"{output_file_prefix}_compression.pdf"
    fig_comp.savefig(comp_file, dpi=300, format="pdf")
    print(f"[info] Compression heatmap saved to {comp_file}")



# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    from pathlib import Path

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    grid_root = repo_root / "paper_results" / "grid_search"
    out_dir = script_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    folder = grid_root / "cifar100" / "deit_tiny_patch16_224.fb_in1k" / "proposal" / "communication=clean"
    df = gather_dataframe(str(folder))
    make_heatmaps(df, str(out_dir / "fig2_heatmap_cifar100"))

    folder = grid_root / "food-101" / "deit_tiny_patch16_224.fb_in1k" / "proposal" / "communication=clean"
    df = gather_dataframe(str(folder))
    make_heatmaps(df, str(out_dir / "fig2_heatmap_food101"))

if __name__ == "__main__":
    main()
