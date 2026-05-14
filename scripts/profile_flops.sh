#!/usr/bin/env bash
# Run tools/compute_flops.py across the compression sweep for every method
# the paper benchmarks. Output lands in paper_results/flops/flops.json.

set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

models=("deit_tiny_patch16_224" "deit_small_patch16_224")

for model in "${models[@]}"; do

    # ---- Base ---------------------------------------------------------------
    python tools/compute_flops.py method=base model="$model"

    # ---- C3-SL --------------------------------------------------------------
    for R in 2 4 8 16 32; do
        python tools/compute_flops.py method=c3_sl method.parameters.R="$R" model="$model"
    done

    # ---- BottleNet++ --------------------------------------------------------
    for compression in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python tools/compute_flops.py method=bottlenet \
            method.parameters.compression="$compression" model="$model"
    done

    # ---- Top-K --------------------------------------------------------------
    for rate in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python tools/compute_flops.py method=top_k \
            method.parameters.rate="$rate" model="$model"
    done

    # ---- Random Top-K -------------------------------------------------------
    for rate in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python tools/compute_flops.py method=random_top_k \
            method.parameters.rate="$rate" model="$model"
    done

    # ---- ADC ----------------------------------------------------------------
    for compression in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python tools/compute_flops.py method=adc \
            method.parameters.compression="$compression" model="$model"
    done

done
