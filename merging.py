# -------------------------------------------------------------
# Imports & small helpers
# -------------------------------------------------------------
from __future__ import annotations

import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from omegaconf import OmegaConf
import hydra

def flatten_params(params):
    if isinstance(params, dict):
        return "_".join(f"{k}={v}" for k, v in params.items())
    return str(params)

def safe_eval(expr: str):
    _safe_globals = {"__builtins__": None, "round": round}
    return eval(expr, _safe_globals, {})

# Register them so that OmegaConf can parse the stored .yaml
OmegaConf.register_new_resolver("eval", safe_eval, replace=True)
OmegaConf.register_new_resolver("flatten_params", flatten_params, replace=True)

def unnorm(batch: torch.Tensor) -> torch.Tensor:
    return batch.cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

def attention_to_heatmap(attn_vec: torch.Tensor, height: int, width: int) -> torch.Tensor:
    attn = attn_vec[1:].reshape(14, 14)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
    attn = F.interpolate(attn[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return attn.cpu()


# An attention class that store the attention matrix 
class Store_Whole_Attn_Wrapper(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.whole_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Normal attention behaviour 
        B, N, C = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)
        q = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # Store class_token attention 
        self.whole_attention = attn.mean(dim=1).detach()

        # Normal attention behaviour 
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x
    

# Apply residual-corrected attention (Abnar & Zuidema, 2020)
def apply_residual(attn: torch.Tensor) -> torch.Tensor:
    B, T, _ = attn.shape
    identity = torch.eye(T, device=attn.device).unsqueeze(0).expand(B, -1, -1)
    return 0.5 * (attn + identity)


def visualise_clusters(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                        output_dir,
                       device: str | torch.device = "cuda") -> None:

    # Get the class names of the dataset 
    class_names = dataloader.dataset.classes

    # Set device 
    device = torch.device(device)
    model.to(device)
    model.train()  # triggers merge in wrapper


    # Apply attention wrapper to first blocks 
    for i in range(2):
        model.model.blocks[i].attn = Store_Whole_Attn_Wrapper(model.model.blocks[i].attn)

    imgs, label = next(iter(dataloader))
    imgs = imgs.to(device)

    with torch.no_grad():
        logits = model(imgs)  # shape [B, 101]
        probs = torch.softmax(logits, dim=-1)  # [B, 101]

    comp = model.compressor_module  # type: ignore[attr-defined]
    if comp is None:
        raise RuntimeError("Model missing compressor_module – cannot proceed.")

    cluster_ids = comp.cluster_ids.cpu()
    first_block_attn = model.model.blocks[0].attn.whole_attention.cpu()
    second_block_attn = model.model.blocks[1].attn.whole_attention.cpu()
    third_block_class_attn = model.model.blocks[2].block.attn.class_token_attention.cpu()  # [B, T]

    def apply_residual(attn: torch.Tensor) -> torch.Tensor:
        B, T, _ = attn.shape
        identity = torch.eye(T, device=attn.device).unsqueeze(0).expand(B, -1, -1)
        return 0.5 * (attn + identity)

    second_block_attn = apply_residual(second_block_attn)
    first_block_attn = apply_residual(first_block_attn)

    imgs_unnorm = unnorm(imgs)
    B, T = third_block_class_attn.shape
    token_keep = max(1, int(comp.token_compression * T))
    
 
    for i in range(max(cluster_ids)):
        idxs = (cluster_ids == i).nonzero(as_tuple=True)[0]

        
        n = len(idxs)
        if n >= 2:
            

            # Compute cluster-level class token attention
            cluster_attn_cls = third_block_class_attn[idxs].mean(dim=0)  # [T]

            # Select top-k tokens from the mean
            topk = torch.topk(cluster_attn_cls[1:], token_keep - 1).indices + 1
            selected = torch.cat([torch.tensor([0]), topk])
            mask = torch.zeros_like(cluster_attn_cls)
            mask[selected] = 1.0
            masked_attn_cls = cluster_attn_cls * mask
            masked_attn_cls = masked_attn_cls / (masked_attn_cls.sum() + 1e-6)
            masked_attn_cls = masked_attn_cls.unsqueeze(0)  # [1, T]


            cluster_probs = probs[i]  # [n, 101]
            cluster_probs = cluster_probs.squeeze(0)  # [101]
            topk = torch.topk(cluster_probs, k=n)
            top_probs = topk.values
            top_indices = topk.indices

            fig, axes = plt.subplots(2, n,
                                        figsize=(2 * n, 5),  # Increase space
                                        gridspec_kw={'wspace': 0.1, 'hspace': 0.1}  # Uniform horizontal & vertical spacing
                                    )


            for col, i in enumerate(idxs):
                img = imgs_unnorm[i]
                H, W = img.shape[1:]

                # Rollou
                attn = masked_attn_cls @ second_block_attn[i] @ first_block_attn[i]  # [1, T]
                heat = attention_to_heatmap(attn.squeeze(0), H, W)

                # Get class name for label[i]
                title = class_names[label[i]]

                # Top row – raw image
                axes[0, col].imshow(img.permute(1, 2, 0).clamp(0, 1))
                axes[0, col].set_xticks([])
                axes[0, col].set_yticks([])
                axes[0, col].set_title(title, fontsize=10)

                # Bottom row – attention overlay
                axes[1, col].imshow(img.permute(1, 2, 0).clamp(0, 1))
                axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

                im = axes[1, col].imshow(heat, cmap="jet", alpha=0.5)


            # pred_text ="Prediction: \n" + "\n".join(f"{class_names[i]}: {p:.2f}" for i, p in zip(top_indices, top_probs))
            # fig.text(0.99, 0.01, pred_text,
            #         ha='right', va='bottom',
            #         fontsize=10, family="monospace",
            #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # correct ="Labels:\n" + "\n".join(class_names[i] for i in label[idxs])
            # fig.text(0.01, 0.01, correct,
            #         ha='left', va='bottom',
            #         fontsize=10, family="monospace",
            #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            
            # Add a single horizontal colorbar below all subplots
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",  pad=0.1, shrink=0.6)
            cbar.set_label("Attention Rollout Intensity")


            plt.savefig(output_dir + f"cluster_{i.item()}_imgs.pdf", dpi=300)
            plt.savefig(output_dir + f"cluster_{i.item()}_imgs.png", dpi=300)
            plt.close()

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
    from omegaconf import OmegaConf

    for model_type in ["tiny"]:
        model_name = "deit_" + model_type + "_patch16_224.fb_in1k" 
        for compression in [0.5]:

            # Match folder naming style exactly (with space!)
            compression_path = "params={'compression'_ " + str(compression) + "}"

            run_dir = f"/home/federico/Desktop/Split_Learning/models/deit_{model_type}_models/results/prova/food-101/{model_name}/proposal/communication=clean/{compression_path}"
            run_cfg = OmegaConf.load(run_dir + "/.hydra/config.yaml")


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