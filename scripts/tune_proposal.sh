#!/usr/bin/env bash
# Grid-search the two ADC hyper-parameters (batch_compression, token_compression).
# Reproduces the data underlying Fig. 2 of the paper.

set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

EXPERIMENT_NAME=${EXPERIMENT_NAME:-grid_search}
DATASET=${DATASET:-cifar_100}
MODEL=${MODEL:-deit_tiny_patch16_224}

batch_compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
token_compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

for batch_compression in "${batch_compressions[@]}"; do
  for token_compression in "${token_compressions[@]}"; do
    python main.py method=proposal \
      "hyperparameters.experiment_name=${EXPERIMENT_NAME}" \
      "dataset=${DATASET}" \
      "model=${MODEL}" \
      method.parameters.token_compression="$token_compression" \
      method.parameters.batch_compression="$batch_compression"
  done
done
