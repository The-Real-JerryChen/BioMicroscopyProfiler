#!/bin/bash

mkdir -p ../logs

methods=("wsl" "mae" "simclr")
ge_tokens=(1 2 4 6 8)
epochs=400

for m in "${methods[@]}"; do
    for g in "${ge_tokens[@]}"; do
        output_name="pretrain-${m}-gt${g}-e${epochs}"
        output_file="../logs/${output_name}.out"
        
        echo "Starting job: ${output_name}"
        
        python pretrain.py \
            --ssl_method ${m} \
            --ge_token ${g} \
            --ge_token_dim 256 \
            --use_wandb \
            --epochs ${epochs} \
            --use_amp \
            --checkpoint_dir ./checkpoints \
            > "${output_file}" 2>&1 &
        
        echo "Started job: ${output_name} (PID: $!)"
        
        sleep 2
    done
done
