#!/usr/bin/env bash
set -euo pipefail

# Get models and datsets 
models=("deit_small_patch16_224" "deit_tiny_patch16_224")

# For each combination of dataset and model 
for model in "${models[@]}"; do

        # ------------ Base ----------- 
    python flops_plot.py\
        method="base"\
        model="$model"


    # ------------ C3-SL -----------
    Rs=(2 4 8 16 32)

    for R in "${Rs[@]}"; do
        python flops_plot.py \
            method.parameters.R="$R" \
            method="c3-sl"\
            model="$model"
    done

    # ------------ Bottlenet -----------
    compressions=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

    for compression in "${compressions[@]}"; do
        python flops_plot.py \
            method.parameters.compression="$compression" \
            method="bottlenet"\
            model="$model"
    done



    # # ------------ Top-k -----------
    # rates=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

    # for rate in "${rates[@]}"; do
    #     python flops_plot.py \
    #         method.parameters.rate="$rate" \
    #         method="top_k"\
    #         model="$model"
    # done


    # # ------------ Random Top-k -----------
    # rates=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

    # for rate in "${rates[@]}"; do
    #     python flops_plot.py \
    #         method.parameters.rate="$rate" \
    #         method="random_top_k"\
    #         model="$model"
    # done

    # ------------ Proposal ----------- 
    compressions=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)

    for compression in "${compressions[@]}"; do
        python flops_plot.py \
            method.parameters.compression="$compression" \
            method="proposal"\
            model="$model"
    done


done
