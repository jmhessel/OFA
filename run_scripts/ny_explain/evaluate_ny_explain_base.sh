#!/bin/bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,2,3,4,5

data=../../datasets/snli_ve_data/snli_ve_dev_5K.tsv
path=../../checkpoints/ofa_base.pt
result_path=../../results/snli_ve
split='snli_ve_zeroshot'
