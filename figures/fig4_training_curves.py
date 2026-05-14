import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
from matplotlib import rcParams
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



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
def load_all_results(root_folders):
    """
    Load and aggregate results from multiple root folders (e.g., prova_51, prova_52...)
    Returns a nested dictionary: results[dataset][model][method] = dict with lists of results across seeds
    """
    methods = ["proposal",  "Top_K", "Random_Top_K", "C3-SL", "Bottlenet", "base", ]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for root_dir in root_folders:
        for dataset in os.listdir(root_dir):
            if dataset not in ["food-101", "cifar100"]:
                continue
            dataset_path = os.path.join(root_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            for model in os.listdir(dataset_path):
                model_path = os.path.join(dataset_path, model)
                if not os.path.isdir(model_path):
                    continue
                for method in methods:
                    method_path = os.path.join(model_path, method, "communication=clean")
                    if not os.path.exists(method_path):
                        continue
                    for param_folder in os.listdir(method_path):
                        if method == "proposal" and "attention" not in param_folder:
                            continue
                        result_file = os.path.join(method_path, param_folder, "final_training_results.json")
                        if not os.path.exists(result_file):
                            result_file = os.path.join(method_path, param_folder, "training_results.json")
                            if not os.path.exists(result_file):
                                continue
                        try:
                            with open(result_file, "r") as f:
                                data = json.load(f)

                            compression = data.get("Compression", None)
                            comm_budget = data.get("Communication cost", [])
                            val_accuracies = data.get("Val accuracies", [])
                            if compression is None or not val_accuracies:
                                continue
                            final_acc = val_accuracies[-1]
                            results[dataset][model][method]["compression"].append(compression)
                            results[dataset][model][method]["final_accuracy"].append(final_acc)
                            results[dataset][model][method]["val_accuracies"].append(val_accuracies)
                            results[dataset][model][method]["communication"].append(comm_budget)
                        except Exception as e:
                            print(f"[ERROR] Failed to load {result_file}: {e}")
    return results

def plot_with_errorbars(all_results, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Color map: proposal gets bright red, others muted
    color_map = {
        "base": "black",               # reference line
        "Random_Top_K": "#1f77b4",     # muted blue
        "Top_K": "#ff7f0e",            # muted orange
        "Bottlenet": "#2ca02c",        # muted green
        "C3-SL": "#9467bd",            # muted purple
        "proposal": "#d62728"          # bright red (highlighted best method)
    }


    method_name = {
        "base": "Base",
        "Random_Top_K": "RandTopK",
        "Top_K": "Top-K",
        "Bottlenet": "BottleNet++",
        "C3-SL": "C3-SL",
        "proposal": r"$\mathbf{ADC}$"
    }

    for dataset in ["food-101", "cifar100"]:
        for model in all_results[dataset]:
            plt.figure(figsize=(10, 6))
            for method, method_data in all_results[dataset][model].items():
                if method == "base":
                    base_accs = method_data["final_accuracy"]
                    if base_accs:
                        mean_base = np.mean(base_accs)
                        plt.axhline(
                            mean_base,
                            color=color_map["base"],
                            linestyle="--",
                            label=method_name["base"],
                            linewidth=1.5
                        )
                    continue

                compressions = np.array(method_data["compression"])
                final_accuracies = np.array(method_data["final_accuracy"])
                unique_compressions = sorted(set(compressions))

                means, stds = [], []
                for c in unique_compressions:
                    if method == "C3-SL" and c == 0.125:
                        if dataset == "cifar100" and model == "deit_small_patch16_224.fb_in1k":
                            accs = [final_accuracies[i]  for i in range(len(compressions)) if compressions[i] == c]
                        else:
                            accs = [final_accuracies[i]  for i in range(len(compressions)) if compressions[i] == c]
                        
                    else:
                        accs = [final_accuracies[i] for i in range(len(compressions)) if compressions[i] == c]
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))

                # Make proposal thicker and fully opaque
                if method == "proposal":
                    lw = 3
                    alpha = 1.0
                    zorder = 5  # bring to front
                else:
                    lw = 1.5
                    alpha = 0.7
                    zorder = 2

                plt.errorbar(
                    unique_compressions,
                    means,
                    yerr=stds,
                    marker='o',
                    label=method_name[method],
                    capsize=4,
                    linewidth=lw,
                    markersize=6,
                    color=color_map[method],
                    alpha=alpha,
                    zorder=zorder
                )

            plt.xlabel("Compression Ratio ξ")
            plt.ylabel("Test Accuracy")
            plt.xlim(0, 0.5)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            model_name = "tiny" if "tiny" in model else "small"
            plot_name = f"{dataset}_{model_name}"
            plt.savefig(os.path.join(output_folder, plot_name + ".pdf"))
            plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_communication_vs_accuracy(
    results,
    dataset,
    model,
    output_folder,
    n_points: int = 300,
    compression_bins = [0.1, 0.15],  # will make plots for [0–0.2], [0.2–0.5]
):
    """
    For each dataset × model × bin range:
    Plot Accuracy vs Communication for all methods that fall into that bin.
    """
    os.makedirs(output_folder, exist_ok=True)

    method_name = {
        "base": "Base",
        "Random_Top_K": "RandTopK",
        "Top_K": "Top-K",
        "Bottlenet": "BottleNet++",
        "C3-SL": "C3-SL",
        "proposal": r"$\mathbf{ADC}$"
    }
    color_map = {
        "base": "black",
        "Random_Top_K": "#1f77b4",
        "Top_K": "#ff7f0e",
        "Bottlenet": "#2ca02c",
        "C3-SL": "#9467bd",
        "proposal": "#d62728"
    }

    # Loop over bins and make one plot per bin
    for b in range(len(compression_bins) - 1):
        low, high = compression_bins[b], compression_bins[b+1]

        # make subfolder for this bin
        bin_folder = os.path.join(output_folder, f"{low}-{high}")
        os.makedirs(bin_folder, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # --- Methods ---
        for method, method_data in results.items():
            comps = np.array([float(c) for c in method_data["compression"]])
            acc_series_all = np.array(method_data["val_accuracies"], dtype=object)
            comm_series_all = np.array(method_data["communication"], dtype=object)

            if not len(comps):
                continue

            # select runs in this bin (always keep base runs with compression > 0.9)
            mask = ((comps >= low) & (comps <= high)) | (comps > 0.9)
            if not mask.any():
                continue

            acc_series = acc_series_all[mask]
            comm_series = comm_series_all[mask]
            selected_comps = comps[mask]

            # drop empties
            filtered = [(c, a) for c, a in zip(comm_series, acc_series) if len(c) and len(a)]
            if not filtered:
                continue

            comm_series, acc_series = zip(*filtered)

            max_comm = max(max(c) for c in comm_series)
            min_comm = min(min(c) for c in comm_series)
            x_line = np.linspace(min_comm, max_comm, n_points)

            interp_acc = []
            for comm, acc in zip(comm_series, acc_series):
                comm = np.array(comm, dtype=float)
                acc  = np.array(acc, dtype=float)
                if len(comm) == 0 or len(acc) == 0:
                    continue
                interp_acc.append(np.interp(x_line, comm, acc))
            interp_acc = np.vstack(interp_acc)

            mean_acc = np.mean(interp_acc, axis=0)
            std_acc = np.std(interp_acc, axis=0)

            # Highlight proposal, keep base dashed maybe
            if method == "proposal":
                lw, alpha, zorder, ls = 3, 1.0, 5, "-"
            elif method == "base":
                lw, alpha, zorder, ls = 2.5, 1.0, 4, "-"
            else:
                lw, alpha, zorder, ls = 1.8, 0.8, 2, "-"

            # compression value (by assumption, only one per bin)
            comp_val = float(selected_comps[0])
            label = f"{method_name[method]} (ξ = {comp_val:.2f})"
            if method == "base":
                label = "Base"

            ax.plot(
                x_line, mean_acc,
                color=color_map[method],
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
                linestyle=ls,
                label=label
            )
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.fill_between(
                x_line, mean_acc-std_acc, mean_acc+std_acc,
                color=color_map[method], alpha=0.15 if method != "base" else 0.05
            )

        ax.set_xlabel("$C_{tot}$")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0.0, 0.9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="lower center",  ncol=2)
        ax.axvline(x=max_comm, color='grey', linestyle='--', linewidth=1)
        ax.text(1.01*max_comm, 0.02, r"$\Gamma$", fontsize=20)
        fig.tight_layout()

        save_name = f"{dataset}_{model}_comm_vs_acc"
        plt.savefig(os.path.join(bin_folder, save_name + ".pdf"))
        plt.close(fig)

# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# from mpl_toolkits.mplot3d.art3d import Line3DCollection

# def plot_3d_accuracy_trajectories(
#     results,
#     dataset,
#     model,
#     output_folder,
# ):
#     """
#     3D plot of Accuracy vs Communication vs Compression, showing full trajectories for all runs.
#     Each run is a line in 3D space.
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     method_name = {
#         "base": "Base",
#         "Random_Top_K": "RandTopK",
#         "Top_K": "Top-K",
#         "Bottlenet": "BottleNet++",
#         "C3-SL": "C3-SL",
#         "proposal": r"$\mathbf{ADC}$"
#     }
#     color_map = {
#         "base": "black",
#         "Random_Top_K": "#1f77b4",
#         "Top_K": "#ff7f0e",
#         "Bottlenet": "#2ca02c",
#         "C3-SL": "#9467bd",
#         "proposal": "#d62728"
#     }

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     for method, method_data in results.items():
#         comps = np.array([float(c) for c in method_data["compression"]])
#         acc_series_all = np.array(method_data["val_accuracies"], dtype=object)
#         comm_series_all = np.array(method_data["communication"], dtype=object)

#         if not len(comps):
#             continue

#         for comp, comm, acc in zip(comps, comm_series_all, acc_series_all):
#             if len(comm) == 0 or len(acc) == 0:
#                 continue

#             # Build a line in 3D: (comm[t], comp, acc[t])
#             xs = np.array(comm, dtype=float)
#             ys = np.full_like(xs, fill_value=comp, dtype=float)
#             zs = np.array(acc, dtype=float)

#             ax.plot(
#                 xs, ys, zs,
#                 color=color_map[method],
#                 linewidth=2.5 if method == "proposal" else 1.2,
#                 alpha=0.9 if method == "proposal" else 0.5,
#                 label=method_name[method] if comp == comps[0] else None
#             )

#     ax.set_xlabel("$C_{tot}$", labelpad=12)
#     ax.set_ylabel("Compression Ratio ξ", labelpad=12)
#     ax.set_zlabel("Test Accuracy", labelpad=12)
#     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
#     ax.legend(loc="upper right")

#     fig.tight_layout()
#     save_name = f"{dataset}_{model}_3D_trajectories"
#     plt.savefig(os.path.join(output_folder, save_name + ".pdf"))
#     plt.close(fig)



def main():
    from pathlib import Path

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    main_root = repo_root / "paper_results" / "main"
    out_dir = script_dir / "outputs" / "fig4"
    out_dir.mkdir(parents=True, exist_ok=True)

    root_dirs = [str(p) for p in sorted(main_root.glob("seed_*"))]
    results = load_all_results(root_dirs)

    plot_with_errorbars(results, output_folder=str(out_dir / "summary"))

    comm_plots_folder = str(out_dir / "communication_vs_accuracy")
    for dataset, models in results.items():
        for model, methods in models.items():
            plot_communication_vs_accuracy(
                results=methods,
                dataset=dataset,
                model=("tiny" if "tiny" in model else "small"),
                output_folder=comm_plots_folder,
            )


if __name__ == "__main__":
    main()