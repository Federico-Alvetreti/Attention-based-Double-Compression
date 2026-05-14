
import numpy as np
import seaborn as sns
import torch 
import matplotlib.pyplot as plt

import os
import hydra
from omegaconf import OmegaConf

def plot_class_cocluster_matrix(M: np.ndarray, class_names: list[str], output_file: str):
    """
    Plot the class co-clustering matrix as a heatmap, showing only the top-k co-clustered classes per class.
    All other values are set to 0 for clarity.
    """
    # Normalize by row
    M_norm = M / (M.sum(axis=1, keepdims=True) + 1e-6)

    # Zero out all but top-k per row
    M_masked = M_norm > 0.005

    M_masked = M_norm * M_masked
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(M_masked, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", square=True, cbar_kws={'label': f'Co-clustering Percentage'})
    plt.title(f"Class Co-Clustering Heatmap")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def compute_class_cocluster_matrix(cluster_ids: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Returns a matrix M where M[i,j] = number of times class i and class j were assigned to the same cluster.
    """
    M = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    for cid in torch.unique(cluster_ids):
        idxs = (cluster_ids == cid).nonzero(as_tuple=True)[0]
        class_ids = labels[idxs].cpu().numpy()

        for i in range(len(class_ids)):
            for j in range(i, len(class_ids)):
                ci, cj = class_ids[i], class_ids[j]
                # M[ci, cj] += 1
                if ci != cj:
                    M[cj, ci] += 1  # symmetry

    return M




def visualise_clusters(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       output_dir,
                       device: str | torch.device = "cuda") -> None:
    
    class_names = dataloader.dataset.classes
    num_classes = len(class_names)
    device = torch.device(device)
    model.to(device)
    model.train()

    # Global co-cluster count matrix
    global_M = np.zeros((num_classes, num_classes), dtype=np.float32)

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        with torch.no_grad():
            _ = model(imgs)

        comp = model.compressor_module  # type: ignore[attr-defined]
        cluster_ids = comp.cluster_ids.cpu()
        labels = labels.cpu()

        # Compute co-clustering *within this batch*
        M_batch = compute_class_cocluster_matrix(cluster_ids, labels, num_classes)
        global_M += M_batch  # Accumulate counts

    # Plot after full pass
    plot_class_cocluster_matrix(global_M, class_names, output_file=os.path.join(output_dir, "class_cocluster_heatmap.png"))


def instantiate_from_config(cfg):

    # Dataset ----------------------------------------------------------------
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=16,
    )

    # Model and pieces -------------------------------------------------------
    backbone_model = hydra.utils.instantiate(cfg.model)

    channel = hydra.utils.instantiate(cfg.communication.channel)


    model = hydra.utils.instantiate(cfg.method.model,
                                    channel=channel,
                                    split_index=cfg.hyperparameters.split_index,
                                    model=backbone_model)
    return model, data_loader




def main():

    # Set seed for reproducibility 
    torch.manual_seed(42) 

    # Select run
    for model_name in ["deit_tiny_patch16_224.fb_in1k", "deit_small_patch16_224.fb_in1k"]:
        for compression in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

            compression_path ="{'compression': "+str(compression) +"}"
            
            # Get directory and congiguration 
            run_dir = "/home/federico/Desktop/Split_Learning/models/deit_tiny_models/results/prova/food-101/"+ model_name + "/proposal/communication=clean/params=" + compression_path
            run_cfg = OmegaConf.load(run_dir + "/.hydra" + "/config.yaml")


            # Load model and data loader 
            model, data_loader = instantiate_from_config(run_cfg)
            model.load_state_dict(torch.load(run_dir + "/best_model.pt", map_location="cpu"))


            output_dir = "/home/federico/Desktop/Split_Learning/plots/clusters/" + model_name +"/" + str(compression) +"/"
            os.makedirs(output_dir, exist_ok=True)

            visualise_clusters(model,
                                data_loader,
                                output_dir,
                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


if __name__ == "__main__":
    main()