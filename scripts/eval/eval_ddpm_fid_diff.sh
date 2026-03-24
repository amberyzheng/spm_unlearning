#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python fid_diff.py --real_path $1 --fake_path $2 