#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
unlearn_percent=$2

c=0
while IFS= read -r forget_idx; do
  forget_idx=$(echo "$forget_idx" | tr ',' ' ')
  echo "trial $c Forgetting: $forget_idx"
  out_file="results/classification/cifar_resnet18_retrain_random/index_${unlearn_percent}/metrics"
  python evaluate.py \
    --model_type classifier \
    --ckpt_path results/classification/cifar_resnet18/1_class/best-epoch=99-val_acc=0.94.ckpt \
    --dataset_name 'cifar10' \
    --support_size_eval 256 \
    --batch_size 256 \
    --gpus 1 \
    --expert_name 'pr' \
    --run_name "eval_cls_cifar_retrain_index${unlearn_percent}" \
    --out_file "$out_file" \
    --support_digits "0,1,2,3,4,5,6,7,8,9" \
    --config config/classification/cifar_resnet18.yaml \
    --seed $c \
    --indexes_to_replace ${forget_idx} \
    --retrain \
    --acc_v2
  c=$((c + 1))
done <  "random_unlearn_idx/random_unlearn_indices_${unlearn_percent}.txt"

