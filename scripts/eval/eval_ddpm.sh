#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

SUPPORT_SIZE=50
for q in {1..9..2}
do
    query_digits="$q"
    support_digits=$(echo {0..9} | tr ' ' '\n' | grep -v "^$q$" | paste -sd, -)
    echo "Query digit: $query_digits"
    echo "Support digits: $support_digits"
    torchrun evaluate.py \
        --config 'config/generation/cifar_ddpm.yaml' \
        --ckpt_path 'results/generation/cifar_ddpm/best-epoch=1999-val_fid=150.17.ckpt' \
        --dataset_name 'cifar10' \
        --model_type 'ddpm' \
        --support_size_eval ${SUPPORT_SIZE} \
        --batch_size 128 \
        --num_classes 10 \
        --support_digits "${support_digits}" \
        --query_digits  $query_digits \
        --no_normalize \
        --seed 42 \
        --total_samples_per_class 5000

    python evaluation/cls_acc_edm.py \
    --query_digits "${query_digits}" \
    --support_digits "${support_digits}" \
    --generated_root "results/images/ddpm/${SUPPORT_SIZE}" \
    --batch_size 128 \
    --device "cuda"
done

