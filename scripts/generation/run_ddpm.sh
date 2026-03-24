#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_P2P_DISABLE=1

python main.py --config config/generation/cifar_ddpm.yaml --seed 0
