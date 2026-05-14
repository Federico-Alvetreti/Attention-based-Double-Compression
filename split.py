#!/usr/bin/env python3
"""
plot.py — Plot (mean ± std) final validation accuracy vs. split point
for each dataset and compression, aggregating across seeds and
including Proposal (ξ), Top-K (rate), and Bottlenet (compression).

Directory structure supported (examples):
  <root>/
    prova_split_10/
    prova_51_split_10/
    prova_114_split_2/
      <dataset>/deit_tiny_patch16_224.fb_in1k/
        proposal/communication=clean/params={'desired_compression'_ 0.25,...}/final_training_results.json
        Top_K/communication=clean/params={'rate'_ 0.5}/final_training_results.json
        Bottlenet/communication=clean/params={'compression'_ 0.25, 'n_layers'_ 2}/final_training_results.json

Outputs one PNG per dataset under --outdir (default: plots/ablation/split_point).

Usage:
  python plot.py --root . --outdir plots/ablation/split_point
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


def get_regexes():
    return {
        "split": re.compile(r"^prova(?:_(\d+))?_split_([3-9]|\d{2,})$"),
        "proposal": re.compile(r"desired_compression'\s*_\s*([0-9.]+|None)"),
        "topk": re.compile(r"rate'\s*_\s*([0-9.]+)"),
        "bottlenet": re.compile(r"compression'\s*_\s*([0-9.]+)"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot final val accuracy vs split point (mean ± std across seeds).")
    p.add_argument("--root", type=str, default=".", help="Root directory containing prova*split* folders.")
    p.add_argument("--outdir", type=str, default="plots/ablation/split_point", help="Where to save generated plots.")
    return p.parse_args()


def parse_seed_and_split(dir_name: str, regex) -> Optional[Tuple[Optional[int], int]]:
    m = regex.match(dir_name)
    if not m:
        return None
    seed_str, split_str = m.groups()
    seed = int(seed_str) if seed_str is not None else None
    return seed, int(split_str)


def parse_float_from_dirname(dirname: str, regex) -> Optional[float]:
    m = regex.search(dirname)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def read_final(json_path: Path) -> Optional[float]:
    try:
        with json_path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return None
    vals = payload.get("Val accuracies", [])
    if not vals:
        return None
    return vals[-1]  # already percentage


def collect_data(root: Path, regexes: Dict) -> Dict[str, Dict[Tuple[str, float], Dict[int, List[float]]]]:
    """
    Collects:
      data[dataset][(method_label, compression)][split_point] = list of values across seeds
    """
    data: Dict[str, Dict[Tuple[str, float], Dict[int, List[float]]]] = {}
    datasets = ["cifar100_128", "food-101_128"]
    model_path = Path("deit_tiny_patch16_224.fb_in1k")

    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        parsed = parse_seed_and_split(split_dir.name, regexes["split"])
        if parsed is None:
            continue
        seed, split_point = parsed

        for dataset in datasets:
            base = split_dir / dataset / model_path
            if not base.exists():
                continue

            # Proposal
            for params_dir in (base / "proposal/communication=clean").glob("params=*"):
                comp = parse_float_from_dirname(params_dir.name, regexes["proposal"])
                if comp is None:
                    continue
                val = read_final(params_dir / "final_training_results.json")
                if val is None:
                    continue
                key = ("Proposal ξ", comp)
                data.setdefault(dataset, {}).setdefault(key, {}).setdefault(split_point, []).append(val)

            # Top-K
            for params_dir in (base / "Top_K/communication=clean").glob("params=*"):
                rate = parse_float_from_dirname(params_dir.name, regexes["topk"])
                if rate is None:
                    continue
                val = read_final(params_dir / "final_training_results.json")
                if val is None:
                    continue
                key = ("Top-K rate", rate)
                data.setdefault(dataset, {}).setdefault(key, {}).setdefault(split_point, []).append(val)

            # Bottlenet
            for params_dir in (base / "Bottlenet/communication=clean").glob("params=*"):
                comp = parse_float_from_dirname(params_dir.name, regexes["bottlenet"])
                if comp is None:
                    continue
                val = read_final(params_dir / "final_training_results.json")
                if val is None:
                    continue
                key = ("Bottlenet rate", comp)
                data.setdefault(dataset, {}).setdefault(key, {}).setdefault(split_point, []).append(val)

    return data

from matplotlib.ticker import FormatStrFormatter

def plot_dataset_by_compression(dataset: str, series: Dict, outdir: Path) -> List[Path]:
    import numpy as np
    import matplotlib.pyplot as plt

    method_colors = {"Proposal ξ": "red", "Top-K rate": "orange", "Bottlenet rate": "green"}
    method_markers = {"Proposal ξ": "o", "Top-K rate": "v", "Bottlenet rate": "s"}
    produced = []

    compressions = sorted({comp for (_m, comp) in series.keys()})
    for comp in compressions:
        plt.figure(figsize=(10,6), dpi=140)
        for method_label in ["Proposal ξ", "Top-K rate", "Bottlenet rate"]:
            key = (method_label, comp)

            if "Proposal" in method_label:
                label_name = r"$\mathbf{ADC}$"
                lw = 3
                alpha = 1.0
                zorder = 5  # bring to front
                    
            elif "Top" in method_label:
                label_name = "Top-K"
                lw = 1.5
                alpha = 0.8
                zorder = 2
            else:
                label_name = "BottleNet++"
                lw = 1.5
                alpha = 0.8
                zorder = 2

            if key not in series:
                continue
            xs_sorted = sorted(series[key].keys())
            ys_mean = [np.mean(series[key][sp]) for sp in xs_sorted]
            ys_std = [np.std(series[key][sp], ddof=0) for sp in xs_sorted]
            plt.errorbar(xs_sorted, ys_mean, yerr=ys_std, capsize=3, marker=method_markers[method_label],
                         linewidth=lw, color=method_colors[method_label], label=label_name, markersize=6,alpha=alpha, zorder=zorder)

        baseline = 0.8255537974683543 if "cifar" in dataset else 0.8118547608437314
        dataset_name = "cifar" if "cifar" in dataset else "food"
        max_y = 0.84 if "cifar" in dataset else 0.82
        min_y = 0.7 if "cifar" in dataset else 0.74
        plt.axhline(y=baseline, color="black", linestyle="--", linewidth=1.5, label="Base")
        plt.ylim(min_y,max_y)
        plt.xlabel("Splitting point")
        plt.ylabel("Test Accuracy")
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{dataset_name}_{comp:g}.pdf"
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        produced.append(outpath)
    return produced


def main():
    args = parse_args()
    regexes = get_regexes()
    root = Path("split_exp").resolve()
    outdir = Path("plots/split").resolve()

    data = collect_data(root, regexes)
    produced: List[Tuple[str, Path]] = []
    for dataset in ["cifar100_128", "food-101_128"]:
        for p in plot_dataset_by_compression(dataset, data.get(dataset, {}), outdir):
            produced.append((dataset, p))

    if not produced:
        print("No plots produced. Check directory structure and file contents.")
    else:
        for ds, p in produced:
            print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()

