#!/usr/bin/env bash
# Drive the full training sweep that produced the numbers in Figures 3 and 4
# of the EUSIPCO 2026 paper: 6 methods x 2 models x 2 datasets x compression
# sweep x 3 seeds.
#
# Output lands in:
#   results/<experiment_name>/<dataset>/<model>/<method>/communication=clean/params=<...>/
#
# Set EXPERIMENT_NAME and SEEDS via environment vars to override the defaults.

set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

EXPERIMENT_NAME=${EXPERIMENT_NAME:-paper}
SEEDS=${SEEDS:-"0 1 2"}

datasets=("cifar_100" "food_101")
models=("deit_tiny_patch16_224" "deit_small_patch16_224")

for seed in $SEEDS; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do

      # Map seed index -> seed value reported in the paper.
      case "$seed" in
        0)   seed_value=43422 ;;
        1)   seed_value=51    ;;
        2)   seed_value=114   ;;
        *)   seed_value="$seed" ;;
      esac

      common=(
        "hyperparameters.experiment_name=${EXPERIMENT_NAME}/seed_${seed}"
        "hyperparameters.seed=${seed_value}"
        "dataset=${dataset}"
        "model=${model}"
      )

      # ---- Base (no compression; one run) ---------------------------------
      python main.py method=base "${common[@]}"

      # ---- BottleNet++ ----------------------------------------------------
      for compression in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python main.py method=bottlenet \
          method.parameters.compression="$compression" "${common[@]}"
      done

      # ---- Top-K ----------------------------------------------------------
      for rate in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python main.py method=top_k \
          method.parameters.rate="$rate" "${common[@]}"
      done

      # ---- Random Top-K ---------------------------------------------------
      for rate in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python main.py method=random_top_k \
          method.parameters.rate="$rate" "${common[@]}"
      done

      # ---- C3-SL ----------------------------------------------------------
      for R in 2 4 8 16 32; do
        python main.py method=c3_sl method.parameters.R="$R" "${common[@]}"
      done

      # ---- ADC -----------------------------------------------------------
      for compression in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python main.py method=adc \
          method.parameters.compression="$compression" "${common[@]}"
      done

    done
  done
done
