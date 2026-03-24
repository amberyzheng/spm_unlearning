#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export NCCL_P2P_DISABLE=1


python main.py \
    --config config/classification/imagenet_resnet18.yaml

