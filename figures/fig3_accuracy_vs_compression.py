# !/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
   'legend.loc': 'lower center'


}
rcParams.update(params)

def load_results(root_dir):

    methods = ["base", "Random_Top_K", "Top_K", "Bottlenet", "C3-SL", "proposal"]
    results = {}
    for method in methods:
        method_path = os.path.join(root_dir, method, "communication=clean")
        results[method] = {"compression": [], "max_val_accuracy": [], "val_accuracies": [], "communication": [], "train_loss": [], "val_loss":[]}

        if not os.path.exists(method_path):
            print(f"[WARNING] Skipping missing method folder: {method_path}")
            continue

        for subdir in os.listdir(method_path):


            if method == "proposal" and not "attention" in subdir:
                print(subdir)
                continue 

            json_path = os.path.join(method_path, subdir, "training_results.json")

            if not os.path.exists(json_path):
                print(f"[WARNING] Missing file: {json_path}")
                continue

            try:

                with open(json_path, "r") as f:
                    data = json.load(f)

                compression = data.get("Compression", None)
                val_accuracies = data.get("Val accuracies", [])
                communication_cost = data.get("Communication cost", [])
                train_loss = data.get("Train losses", [])
                val_loss = data.get("Val losses", [])

                if compression is None or  val_accuracies == [] or communication_cost == []:
                    print(f"[WARNING] Missing data in {json_path}")
                    continue
                
                if compression > 0.6 and method != "base":
                    continue
 
                max_val_acc = val_accuracies[-1]
                results[method]["communication"].append(communication_cost)
                results[method]["compression"].append(compression)
                results[method]["max_val_accuracy"].append(max_val_acc)
                results[method]["val_accuracies"].append(val_accuracies)
                results[method]["train_loss"].append(train_loss)
                results[method]["val_loss"].append(val_loss)

            except Exception as e:
                print(f"[ERROR] Failed to process {json_path}: {e}")

    return results


def plot_results(results, output_path):

    dataset= output_path.split("/")[-2]
    model = output_path.split("/")[-1]
    plt.figure(figsize=(10, 6))

    print(f"model: {model} \ndataset: {dataset}")
    for method, data in results.items():

        if method == "base":
            # Plot a horizontal line at the base accuracy
            base_acc = data["max_val_accuracy"][0]
            plt.hlines(base_acc, xmin=0, xmax=0.65, color ="black", linestyles="--", label="base (no compression)", linewidth=2)
        else:
            if data["compression"] != []:
                sorted_data = sorted(zip(data["compression"], data["max_val_accuracy"]))
                x_sorted, y_sorted = zip(*sorted_data)

            plt.plot(x_sorted, y_sorted, marker="o", label=method, linewidth=2)

    
    plt.xlim(0, 0.5)
    plt.xlabel("Compression ")
    plt.ylabel("Final Test Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path_pdf = output_path + "/" + "accuracy_vs_compression.pdf"
    plt.savefig(save_path_pdf)
    plt.close()



def plot_best_envelopes(
    results,
    output_path,
    n_points: int = 400,
    max_compression: float = 0.30,   # threshold for runs to keep
):
    """
    Draw an upper-envelope Accuracy-vs-Communication plot for each method
    (excluding the 'base' run).  
    **Only runs whose compression ≤ `max_compression` are considered.**

    Parameters
    ----------
    results : dict
        Nested results dictionary with keys:
            results[method]['communication']  -> list[list[float]]
            results[method]['val_accuracies'] -> list[list[float]]
            results[method]['compression']    -> list[float]
            results['base']                   -> single-run baseline
    output_path : str
        Folder where the figure will be written.
        The parent folder name is interpreted as the dataset,
        the leaf folder name as the model.
    n_points : int
        Number of x-coordinates used when forming the envelope.
    max_compression : float
        Compression threshold – runs with compression > threshold are ignored.
    """
    # ------------------------------------------------------------------ #
    # 1.  Basic book-keeping                                             #
    # ------------------------------------------------------------------ #


    methods = [m for m in results if m != "base"]
    if not methods:
        print("[WARNING] No methods to plot.")
        return

    # ------------------------------------------------------------------ #
    # 2.  Pre-filter runs by compression                                 #
    # ------------------------------------------------------------------ #
    filtered = {}                         # store per-method filtered data
    global_comprs = []                    # used to build a consistent colour-bar

    for method in methods:
        comps = np.asarray(results[method]["compression"])
        keep  = np.where(comps <= max_compression)[0]

        if keep.size == 0:
            print(f"[INFO] Method '{method}' skipped – no runs below "
                  f"max_compression={max_compression}.")
            continue

        comm  = [results[method]["communication"][i]   for i in keep]
        acc   = [results[method]["val_accuracies"][i]  for i in keep]
        comps = comps[keep]

        filtered[method] = dict(
            communication=comm,
            val_accuracies=acc,
            compression=comps,
        )
        global_comprs.extend(comps.tolist())

    if not filtered:
        print("[WARNING] Nothing to plot after compression filtering.")
        return

    # ------------------------------------------------------------------ #
    # 3.  Set up figure / colour mapping                                 #
    # ------------------------------------------------------------------ #
    cmin, cmax = min(global_comprs), max(global_comprs)
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    cmap = cm.get_cmap("viridis")

    n_cols = len(filtered)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(6 * n_cols, 6), sharey=True,
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = [axes]

    # Baseline values (identical for every subplot)
    base_comm = results["base"]["communication"][0]
    base_acc  = results["base"]["val_accuracies"][0]

    # ------------------------------------------------------------------ #
    # 4.  Per-method envelope computation                                #
    # ------------------------------------------------------------------ #
    for ax, (method, data) in zip(axes, filtered.items()):
        comm_series  = [ [0] + c for c in data["communication"] ]
        acc_series   = [ [0] + a for a in data["val_accuracies"] ]
        compressions = data["compression"]

        max_comm = max(max(c) for c in comm_series)
        x_line   = np.linspace(0.0, max_comm, n_points)

        # Interpolate all runs onto the common x-axis
        interp_acc = np.empty((len(compressions), n_points))
        for i, (comm, acc) in enumerate(zip(comm_series, acc_series)):
            interp_acc[i] = np.interp(x_line, comm, acc)

        best_idx          = np.argmax(interp_acc, axis=0)
        best_acc          = interp_acc[best_idx, np.arange(n_points)]
        best_compressions = compressions[best_idx]

        # Create coloured line segments
        points   = np.column_stack([x_line, best_acc])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colours  = cmap(norm(best_compressions[:-1]))
        lc = LineCollection(segments, colors=colours, linewidths=4.0)
        ax.add_collection(lc)

        # Optionally mark the baseline
        # ax.plot(base_comm, base_acc, marker="x", color="red", zorder=3)

        ax.set_xlim(0, max_comm)
        ax.set_xlabel("Communication")
        ax.set_title(method)
        ax.grid(True)

    axes[0].set_ylabel("Best Validation Accuracy")


    # Shared colour-bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes, orientation="horizontal", pad=0.08,
        label="Compression"
    )

    # ------------------------------------------------------------------ #
    # 5.  Save and exit                                                 #
    # ------------------------------------------------------------------ #
    save_path_png = os.path.join(output_path, "best_envelope_all_methods.png")
    save_path_pdf = os.path.join(output_path, "best_envelope_all_methods.pdf")
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close(fig)

def plot_commmunication_without_compression(
    results,
    output_path,
    n_points: int = 400,
    max_compression: float = 0.2,   # ← NEW: threshold for runs to keep
):
    """
    Overlay the best-envelope Accuracy-vs-Communication curves for each method
    (excluding 'base') on a single set of axes.

    Only runs whose compression ≤ `max_compression` are considered.
    """
    # ------------------------------------------------------------------ #
    # 1.  Basic book-keeping                                             #
    # ------------------------------------------------------------------ #
    dataset = output_path.split("/")[-2]
    model   = output_path.split("/")[-1]

    methods = [m for m in results if m != "base"]
    if not methods:
        print("[WARNING] No methods to plot.")
        return

    colour_cycle = plt.get_cmap("tab10").colors   # up to 10 distinct colours
    fig, ax = plt.subplots(figsize=(9, 6))

    # Baseline (always plotted)
    base_comm = results["base"]["communication"][0]
    base_acc  = results["base"]["val_accuracies"][0]
    ax.plot(
        base_comm, base_acc,
        marker="x", color="black", linewidth=2, label="1 (base)"
    )

    # ------------------------------------------------------------------ #
    # 2.  Per-method envelope computation *after* compression filtering  #
    # ------------------------------------------------------------------ #
    plotted_any = False
    for idx, method in enumerate(methods):
        comps = np.asarray(results[method]["compression"])
        keep  = np.where(comps <= max_compression)[0]

        if keep.size == 0:                      # skip if nothing qualifies
            print(f"[INFO] Method '{method}' skipped – no runs below "
                  f"max_compression={max_compression}.")
            continue

        comm_series = [ [0] + results[method]["communication"][i]  for i in keep ]
        acc_series  = [ [0] + results[method]["val_accuracies"][i] for i in keep ]

        # Common x-grid up to the largest communication budget observed
        max_comm = max(max(c) for c in comm_series)
        x_line   = np.linspace(0.0, max_comm, n_points)

        # Interpolate each qualifying run
        interp_acc = np.empty((len(keep), n_points))
        for i, (comm, acc) in enumerate(zip(comm_series, acc_series)):
            interp_acc[i] = np.interp(x_line, comm, acc)

        best_acc = interp_acc.max(axis=0)

        colour = colour_cycle[idx % len(colour_cycle)]
        ax.plot(
            x_line, best_acc, color=colour, linewidth=2.2,
            label=method
        )
        plotted_any = True

    if not plotted_any:
        print("[WARNING] Nothing to plot after compression filtering.")
        plt.close(fig)
        return

    # ------------------------------------------------------------------ #
    # 3.  Aesthetics                                                     #
    # ------------------------------------------------------------------ #
    ax.set_xlim(left=0)
    ax.set_xlabel("Communication")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax.set_ylabel("Best Validation Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_path_png = os.path.join(output_path, "best_envelope_all_method_no_comps.png")
    save_path_pdf = os.path.join(output_path, "best_envelope_all_method_no_comps.pdf")
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close(fig)


def plot_communication(results, output_path):
    """
    Plot communication vs. accuracy for each method in separate plots.
    
    Args:
        results (dict): Dictionary containing method results.
        output_path (str): Directory to save the plots.
    """
    compressions_to_plot = {}

    methods = [m for m in results if m != "base"]

    # Compute global min/max compression for consistent color mapping
    all_compressions = [
        c for m in methods
        for c in (compressions_to_plot.get(m) or results[m]["compression"])
    ]
    global_min = min(all_compressions)
    global_max = max(all_compressions)

    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
    cmap = cm.get_cmap("viridis")

    for method in methods:
        data = results[method]
        allowed = compressions_to_plot.get(method, None)

        for metric in ["val_accuracies"]:

            if allowed is None:
                filtered = list(zip(data["compression"], data["communication"], data[metric]))
            else:
                allowed_set = set(allowed)
                filtered = [
                    (c, comm, acc)
                    for c, comm, acc in zip(data["compression"], data["communication"], data[metric])
                    if c in allowed_set
                ]

            if not filtered:
                print(f"[WARNING] No matching compressions for method '{method}'")
                continue

            compressions, communications, accuracies = zip(*filtered)
            sorted_indices = np.argsort(compressions)
            min_accuracy = 1
            fig, ax = plt.subplots(figsize=(10, 6))
            for idx in sorted_indices:
                compression = compressions[idx]
                communication = communications[idx]
                val_accuracy = accuracies[idx]
                color = cmap(norm(compression))
                ax.plot(communication, val_accuracy, marker="o", label=f"ξ={compression:.3f}",
                        color=color, linewidth=2, alpha=0.7)
                min_accuracy = min(min_accuracy, min(val_accuracy))

            # Plot base
            base_comm = results["base"]["communication"][0]
            base_acc = results["base"][metric][0]
            ax.plot(base_comm, base_acc, marker="x", label="base", color="red", linewidth=2)

            ax.axvline(x=750000, color='black', linestyle='--', linewidth=1)
            ax.text(760000, min_accuracy+0.05, r"$\Gamma$", fontsize=20)


            ax.set_xlabel("$C_{tot}$")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.set_ylabel("Test Accuracy")
            ax.grid(True)
            ax.legend()

            plt.tight_layout()

            # Save figure
            filename_base = f"{method}_{metric}_vs_communication"
            plt.savefig(os.path.join(output_path, filename_base + ".pdf"))
            plt.close()


def main():
    from pathlib import Path

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    out_root = script_dir / "outputs" / "fig3"
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = ["cifar100", "food-101"]
    models = ["deit_small_patch16_224.fb_in1k", "deit_tiny_patch16_224.fb_in1k"]
    seeds = sorted((repo_root / "paper_results" / "main").glob("seed_*"))

    for seed_dir in seeds:
        for model in models:
            for dataset in datasets:
                results_path = seed_dir / dataset / model
                if not results_path.exists():
                    continue
                plot_path = out_root / seed_dir.name / dataset / model
                plot_path.mkdir(parents=True, exist_ok=True)

                results = load_results(str(results_path))
                plot_results(results, str(plot_path))
                plot_communication(results, str(plot_path))


if __name__ == "__main__":
    main()

