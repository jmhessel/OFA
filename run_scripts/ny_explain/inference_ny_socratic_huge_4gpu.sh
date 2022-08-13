#!/bin/bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7061


log_dir=./logs_huge
save_dir=./checkpoints_huge
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

# socratic_split=0_test.tsv
data_dir=/home/jackh/caption-this/experiments/OFA_tsv

restore_file=../../checkpoints/ofa_huge.pt
selected_cols=0,2,3,4,5

task=ny_explain
arch=ofa_huge
batch_size=16

max_src_length=80
max_tgt_length=128
max_src_length=80
max_tgt_length=128
num_bins=1000
patch_image_size=480
prompt_type="prev_output"

echo "hi"

for split in {0,1,2,3,4}; do
    for tr_split in {val,test,train}; do
	data=${data_dir}/socratic_split\=${split}_${tr_split}.tsv,${data_dir}/socratic_split\=${split}_${tr_split}.tsv,${data_dir}/socratic_split\=${split}_${tr_split}.tsv

	if [ $split == 4 ]; then
	    checkpoint_path="checkpoints_huge/7_5e-5_4/checkpoint.best_loss_3.9500.pt"
	elif [ $split == 3 ]; then
	    checkpoint_path="checkpoints_huge/7_5e-5_3/checkpoint.best_loss_3.9410.pt"
	elif [ $split == 2 ]; then
	    checkpoint_path="checkpoints_huge/7_5e-5_2/checkpoint.best_loss_3.9350.pt"
	elif [ $split == 1 ]; then
	    checkpoint_path="checkpoints_huge/7_5e-5_1/checkpoint.best_loss_4.0130.pt"
	elif [ $split == 0 ]; then
	    checkpoint_path="checkpoints_huge/7_5e-5_0/checkpoint.best_loss_3.9740.pt"
	fi
	
	save_path=${save_dir}/${split}_socratic_inference
	mkdir -p $save_path

	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
		${data} \
		--path=${checkpoint_path} \
		--user-dir=${user_dir} \
		--bpe-dir=${bpe_dir} \
		--selected-cols=${selected_cols} \
		--task=ny_explain \
		--patch-image-size=${patch_image_size} \
		--max-src-length=80 \
		--batch-size=${batch_size} \
		--log-format=simple --log-interval=10 \
		--seed=7 \
		--max-src-length=${max_src_length} \
		--max-tgt-length=${max_tgt_length} \
		--gen-subset=${tr_split} \
		--results-path=${save_path} \
		--zero-shot \
		--prompt-type='prev_output' \
		--fp16 \
		--num-workers=0
    done;
done;

# actually, looks like sampler doesn't support sampling, just do beam, its fine.
# --eval-args='{"nbest": 10, "sampling": true, "topp": 0.95}'
