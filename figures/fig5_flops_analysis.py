#!/usr/bin/env python3
"""Plot edge and server FLOP overheads for Proposal vs Bottlenet."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_FILE = REPO_ROOT / "paper_results" / "flops" / "flops.json"
PLOTS_DIR = SCRIPT_DIR / "outputs"
PLOT_FILENAME_EDGE = "edge_overhead.pdf"
PLOT_FILENAME_SERVER = "server_usage.pdf"
TARGET_MODELS = [
    "deit_tiny_patch16_224.fb_in1k",
]
MODEL_DISPLAY_NAMES = {
    "deit_small_patch16_224.fb_in1k": "DeiT-S",
    "deit_tiny_patch16_224.fb_in1k": "DeiT-T",
}
TARGET_METHODS = {
    "proposal": r"$\bf{ADC}$",
    "Bottlenet": "BottleNet++",
    "C3-SL": "C3-SL",
}
METHOD_COLORS = {
    r"$\bf{ADC}$": "red",
    "BottleNet++": "green",
    "C3-SL": "purple",
}

def load_flops(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find FLOP results at {path}")
    with path.open("r") as handle:
        return json.load(handle)

def prepare_series(model_blob: Dict) -> Dict[str, List[Tuple[float, float, float]]]:
    baseline = model_blob.get("base", {}).get("1")
    if baseline is None:
        raise KeyError("Missing base -> 1 entry needed for overhead computation")
    edge_base = baseline["edge_flops"]
    server_base = baseline["server_flops"]
    series = {}
    for method_key, label in TARGET_METHODS.items():
        method_blob = model_blob.get(method_key)
        if method_blob is None:
            continue
        points = []
        for comp_str, metrics in method_blob.items():
            compression = float(comp_str)
            edge_pct = 100 * metrics["edge_flops"] / edge_base
            server_pct = 100 * metrics["server_flops"] / server_base
            points.append((compression, edge_pct, server_pct))
        points.sort(key=lambda item: item[0])
        if points:
            series[label] = points
    if not series:
        raise KeyError("None of the target methods were found for this model")
    return series

def plot_single_metric(
    method_series: Dict[str, List[Tuple[float, float, float]]],
    metric_idx: int,
    ylabel: str,
    baseline: float,
    save_path: Optional[Path] = None,
    break_range: Optional[Tuple[float, float]] = None,
) -> None:
    
    if break_range:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 4), 
                                       gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.1})
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=(3.5, 4))
        axes = [ax]
        ax1, ax2 = ax, None

    for ax_curr in axes:
        # Reference line
        ax_curr.axhline(baseline, color="black", linewidth=1.2, linestyle="--", label="Base")
        
        for label, points in method_series.items():
            compressions = [pt[0] for pt in points]
            vals = [pt[metric_idx] for pt in points]
            color = METHOD_COLORS.get(label, "blue")
            ax_curr.plot(compressions, vals, marker="o", markersize=3, label=label, color=color)

    if break_range:
        break_start, break_end = break_range
        
        # Configure Top Axis
        ax1.set_ylim(break_end, 103) 
        ax1.set_yticks([100, 102])
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(labeltop=False, bottom=False) 
        
        # Configure Bottom Axis
        ax2.set_ylim(0, break_start)
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.tick_bottom()
        
        # --- DRAW DOTTED LINES AT THE CORNERS ---
        # d controls how long the dotted line segment is
        d = 0.05 
        kwargs = dict(color='k', linestyle=':', linewidth=1, clip_on=False)
        
        # Top Plot (ax1): Draw dotted lines downwards from the bottom corners
        # (0,0) is bottom-left, (1,0) is bottom-right of ax1
        ax1.plot([0, 0], [0, -d], transform=ax1.transAxes, **kwargs)
        ax1.plot([1, 1], [0, -d], transform=ax1.transAxes, **kwargs)

        # Bottom Plot (ax2): Draw dotted lines upwards from the top corners
        # (0,1) is top-left, (1,1) is top-right of ax2
        ax2.plot([0, 0], [1, 1+d], transform=ax2.transAxes, **kwargs)
        ax2.plot([1, 1], [1, 1+d], transform=ax2.transAxes, **kwargs)

        # Labels & Legend
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
        ax2.set_xlabel("Compression Ratio ξ")
        ax2.legend(loc="best")
        
    else:
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Compression Ratio ξ")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.05))

    if break_range:
        plt.subplots_adjust(left=0.18, bottom=0.12, right=0.95, top=0.95)
    else:
        fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.close(fig)

def main() -> None:
    raw_data = load_flops(RESULTS_FILE)
    
    target_model = TARGET_MODELS[0]
    model_blob = raw_data.get(target_model)
    if model_blob is None:
        raise SystemExit(f"Model {target_model} not found in results.")
        
    series_data = prepare_series(model_blob)
    
    # 1. Edge Total Computation
    plot_single_metric(
        series_data,
        metric_idx=1,
        ylabel="Client Total FLOPs (%)",
        baseline=100.0,
        save_path=PLOTS_DIR / PLOT_FILENAME_EDGE
    )
    
    # 2. Server Usage (with dotted break)
    plot_single_metric(
        series_data,
        metric_idx=2,
        ylabel="Server Total FLOPs (%)",
        baseline=100.0,
        save_path=PLOTS_DIR / PLOT_FILENAME_SERVER,
        break_range=(65, 99)
    )

if __name__ == "__main__":
    main()