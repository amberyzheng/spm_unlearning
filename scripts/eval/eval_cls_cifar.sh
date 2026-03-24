#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

num_unlearn_class=$2


if [ "$num_unlearn_class" -eq 5 ]; then
  while IFS= read -r query_digits; do
    support_digits=$(echo {0..9} | tr ' ' '\n' | grep -v -E "^($(echo "$query_digits" | sed 's/,/|/g'))$" | paste -sd, -)
    out_file="results/classification/cifar_resnet18_retrain/5_class/metrics_${query_digits//,/}"
    echo "Using query_digits: $query_digits"
    echo "Using support_digits: $support_digits"
    python evaluate.py \
        --model_type classifier \
        --ckpt_path results/classification/cifar_resnet18/1_class/best-epoch=99-val_acc=0.94.ckpt \
        --dataset_name 'cifar10' \
        --support_size_eval 256 \
        --batch_size 256 \
        --gpus 1 \
        --expert_name 'pr' \
        --run_name "eval_cls_cifar_5c_${query_digits//,/}" \
        --out_file "$out_file" \
        --support_digits "$support_digits" \
        --query_digits "$query_digits" \
        --retrain \
        --config config/classification/cifar_resnet18.yaml \
        --acc_v2
  done < "combo.txt"
  exit 0
elif [ "$num_unlearn_class" -eq 1 ]; then
  for q in {0..9}; do
    query_digits="$q"
    support_digits=$(echo {0..9} | tr ' ' '\n' | grep -v "^$q$" | paste -sd, -)
    out_file="results/classification/cifar_resnet18_retrain/1_class/metrics"
    python evaluate.py \
        --model_type classifier \
        --ckpt_path results/classification/cifar_resnet18/1_class/best-epoch=99-val_acc=0.94.ckpt \
        --dataset_name 'cifar10' \
        --support_size_eval 256 \
        --batch_size 256 \
        --gpus 1 \
        --expert_name 'pr' \
        --run_name "eval_cls_cifar_q${q}" \
        --out_file "$out_file" \
        --support_digits "$support_digits" \
        --query_digits "$query_digits" \
        --retrain \
        --config config/classification/cifar_resnet18.yaml \
        --acc_v2 
  done
  exit 0

fi