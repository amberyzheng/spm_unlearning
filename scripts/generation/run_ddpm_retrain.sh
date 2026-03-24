#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export NCCL_P2P_DISABLE=1
for q in {1..9..2}; do
    query_digits="$q"
    python main_retrain.py \
        --config config/generation/cifar_ddpm.yaml \
        --seed 0 \
        --query_digits "$query_digits"
done
