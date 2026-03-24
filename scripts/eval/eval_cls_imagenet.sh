#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

for c in {0..9}; do
    query_classes="$c"
    # support is all other classes
    support_classes=$(seq 0 999 | grep -v "^$c\$" | paste -sd, -)
    echo "Using query_classes: $query_classes"
    echo "Using support_classes: $support_classes"

    out_file="results/classification/imagenet_resnet18_retrain/metrics"
    echo "Query class: $query_classes"
    echo "Support classes count: $(echo "$support_classes" | tr ',' '\n' | wc -l)"
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 evaluate.py \
        --model_type classifier \
        --ckpt_path results/classification/imagenet_resnet18/best-epoch=159-val_acc=0.52.ckpt \
        --dataset_name 'imagenet' \
        --support_size_eval 256 \
        --batch_size 1024 \
        --gpus 1 \
        --expert_name 'pr' \
        --run_name "eval_cls_imagenet_c${c}" \
        --datadir './data/imagenet' \
        --num_classes 1000 \
        --support_digits "$support_classes" \
        --query_digits "$query_classes" \
        --out_file "$out_file" \
        --retrain \
        --config config/classification/imagenet_resnet18.yaml \
        --acc_v2 
done

