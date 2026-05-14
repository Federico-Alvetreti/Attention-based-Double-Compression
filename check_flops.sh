#!/usr/bin/env bash
set -euo pipefail

# ------------ Proposal ----------- 
compressions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

for compression in "${compressions[@]}"; do
    python main.py \
        method.parameters.compression="$compression" 
done