# Communication Efficient Split Learning of ViTs with Attention-based Double Compression

Official code for the EUSIPCO 2026 paper *"Communication Efficient Split Learning
of ViTs with Attention-based Double Compression"* by **Federico Alvetreti, Jary
Pomponi, Paolo Di Lorenzo, Simone Scardapane**.

The paper introduces **ADC**, a dual-stream compression strategy for Split
Learning of Vision Transformers. ADC (i) clusters batch samples that share
similar class-token attention patterns and replaces each cluster by its
average activation, and (ii) prunes the merged activations down to the
top-$k$ tokens identified by the cluster centroid. The two steps act on the
batch and token dimensions respectively, so the total compression ratio
factorises as $\xi = (T/B)\cdot(k/n)$.

## Abstract

> Split Learning (SL) enables collaborative training across edge devices and
> cloud servers, yet its practical deployment is often hindered by the
> substantial communication overhead required to transmit intermediate
> activations. This bottleneck is particularly severe in Vision Transformers
> (ViTs) due to their high-dimensional token representations. While existing
> compression techniques address this, they typically apply uniform reduction
> across features, leading to significant performance degradation under strict
> bandwidth constraints. To overcome this limitation, we introduce
> Attention-based Double Compression (ADC), a communication-efficient SL
> framework tailored for ViTs. ADC employs a dual-stream compression strategy:
> it merges batch samples exhibiting similar attention patterns and
> selectively retains only the most informative tokens within these merged
> representations. This inherent dimensionality reduction simultaneously
> lowers bandwidth demands, accelerates training, and enables intrinsic
> gradient compression, allowing for efficient end-to-end learning without
> additional tuning or approximation. Experimental results demonstrate that
> ADC significantly outperforms state-of-the-art methods, maintaining high
> accuracy even at extreme compression regimes.

## Installation

```bash
conda env create -f environment.yaml
conda activate split-learning
```

The environment pins PyTorch 2.5 + CUDA 12, timm 1.0.14, Hydra 1.3, and the
`kmeans_pytorch` clustering helper used by ADC.

## Data

```bash
python scripts/download_data.py
```

This downloads CIFAR-100 and Food-101 into `./data/` (via `torchvision`) and
caches the pretrained DeiT-T / DeiT-S weights through `timm`.

## Quickstart

Train ADC on CIFAR-100 with DeiT-T at target compression $\xi=0.1$:

```bash
python main.py method=adc dataset=cifar_100 model=deit_tiny_patch16_224 \
    method.parameters.compression=0.1
```

Output is written under
`results/<experiment_name>/<dataset>/<model>/<method>/communication=clean/params=<...>/`
including `training_results.json`, `best_model.pt`, and Hydra metadata. The
default `experiment_name` is `default`; override with
`hyperparameters.experiment_name=<name>`.

## Per-method usage

The compression sweep below matches the values reported in the paper. All
runs use `dataset={cifar_100, food_101}`, `model={deit_tiny_patch16_224,
deit_small_patch16_224}`, and split point `l=3` (default).

### Base (no compression)

```bash
python main.py method=base
```

### BottleNet++

```bash
for c in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
  python main.py method=bottlenet method.parameters.compression=$c
done
```

### Top-K

```bash
for k in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
  python main.py method=top_k method.parameters.rate=$k
done
```

### Random Top-K

```bash
for k in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
  python main.py method=random_top_k method.parameters.rate=$k
done
```

### C3-SL

```bash
for R in 2 4 8 16 32; do
  python main.py method=c3_sl method.parameters.R=$R
done
```

### ADC

```bash
for c in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
  python main.py method=adc method.parameters.compression=$c
done
```

ADC also exposes the two underlying factors directly if you want to step off
the diagonal $T/B = k/n = \sqrt{\xi}$:

```bash
python main.py method=adc \
    method.parameters.batch_compression=0.4 \
    method.parameters.token_compression=0.6
```

## How the paper results were produced

Numbers in Figures 3 and 4 come from the full Hydra sweep wrapped in
[`scripts/run_paper_experiments.sh`](scripts/run_paper_experiments.sh): six
methods × two models × two datasets × the compression sweep above × three
seeds (43422, 51, 114). FLOP measurements behind Figure 5 are produced by
[`scripts/profile_flops.sh`](scripts/profile_flops.sh), which drives
[`tools/compute_flops.py`](tools/compute_flops.py). The $(T/B, k/n)$ grid
search visualised in Figure 2 was run with
[`scripts/tune_adc.sh`](scripts/tune_adc.sh).

The corresponding numeric outputs are archived locally; each plotting script
reads from there.

## Repository layout

```
.
├── main.py                       Training entry point (Hydra)
├── configs/                      Hydra config groups
│   ├── default.yaml
│   ├── communication/{clean,noisy}.yaml
│   ├── dataset/{cifar_100,food_101}.yaml
│   ├── method/{base,bottlenet,c3_sl,top_k,random_top_k,adc}.yaml
│   ├── model/{deit_tiny,deit_small}_patch16_224.yaml
│   └── optimizer/adam.yaml
├── methods/                      Compression methods (paper baselines + ADC)
├── comm/communication.py         Clean / Gaussian-noise channels
├── scripts/
│   ├── download_data.py
│   ├── run_paper_experiments.sh
│   ├── profile_flops.sh
│   ├── tune_adc.sh
│   └── slurm/                    CINECA / Leonardo job templates
└── tools/compute_flops.py        FLOP profiler
```

## Citation

```bibtex
@inproceedings{alvetreti2026adc,
  title     = {Communication Efficient Split Learning of {ViTs} with
               Attention-based Double Compression},
  author    = {Alvetreti, Federico and Pomponi, Jary and
               Di Lorenzo, Paolo and Scardapane, Simone},
  booktitle = {Proceedings of the 34th European Signal Processing Conference
               (EUSIPCO)},
  year      = {2026},
}
```

## Acknowledgements

This work has been supported by:

1. SNS JU project 6G-GOALS under the EU's Horizon program Grant Agreement
   No 101139232.
2. Sapienza grant RG123188B3EF6A80 (CENTS), by European Union under the
   Italian National Recovery and Resilience Plan of NextGenerationEU,
   partnership on Telecommunications of the Future (PE00000001 — program
   RESTART).
3. "Sapienza, Avvio alla ricerca" grant (UGOV 1201260).

We also acknowledge ISCRA for awarding this project access to the LEONARDO
supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA
(Italy).

Corresponding author: <federico.alvetreti@uniroma1.it>
