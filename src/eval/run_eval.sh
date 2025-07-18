#!/bin/bash

mkdir -p ./logs

python main.py --method wsl --ge_token 1 --ge_token_dim 256 --use_wandb --supervised_weight 0.01

# python main.py --method simclr --ge_token 4 --ge_token_dim 256 --use_wandb --supervised_weight 0.0001 --loss lap

# python main.py --method mocov3 --ge_token 4 --ge_token_dim 256 --use_wandb --supervised_weight 1e-5 --loss lap

# python main.py --method mae --ge_token 8 --ge_token_dim 256 --use_wandb --supervised_weight 1e-4 --loss con
