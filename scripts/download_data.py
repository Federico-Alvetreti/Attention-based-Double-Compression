"""Download the datasets and pretrained models used by the EUSIPCO 2026 paper.

Run from the repo root::

    python scripts/download_data.py

Datasets go to ./data/ (CIFAR-100, Food-101). Pretrained DeiT weights are
cached by timm in its default cache directory (~/.cache/huggingface).
"""

import os

import timm
import torchvision


def main():
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_root, exist_ok=True)

    torchvision.datasets.CIFAR100(data_root, train=True, download=True)
    torchvision.datasets.CIFAR100(data_root, train=False, download=True)

    torchvision.datasets.Food101(data_root, split="train", download=True)
    torchvision.datasets.Food101(data_root, split="test", download=True)

    timm.create_model(model_name="deit_small_patch16_224.fb_in1k", pretrained=True)
    timm.create_model(model_name="deit_tiny_patch16_224.fb_in1k", pretrained=True)


if __name__ == "__main__":
    main()
