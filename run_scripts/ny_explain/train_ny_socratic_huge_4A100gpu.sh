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
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
warmup_ratio=0.06
batch_size=8
update_freq=1

resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1

dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=128
num_bins=1000
patch_image_size=480
prompt_type="prev_output"

echo "hi"


for split in {4,3,2,1,0}; do
    for max_epoch in {7,}; do
	echo "max_epoch "${max_epoch}
	for lr in {5e-5,}; do
	    echo "lr "${lr}

	    data=${data_dir}/socratic_split\=${split}_train.tsv,${data_dir}/socratic_split\=${split}_val.tsv
	    
	    log_file=${log_dir}/${max_epoch}"_"${lr}"_"${split}".log"
	    save_path=${save_dir}/${max_epoch}"_"${lr}"_"${split}
	    mkdir -p $save_path

	    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../train.py \
				$data \
				--selected-cols=${selected_cols} \
				--bpe-dir=${bpe_dir} \
				--user-dir=${user_dir} \
				--restore-file=${restore_file} \
				--reset-optimizer --reset-dataloader --reset-meters \
				--save-dir=${save_path} \
				--task=${task} \
				--arch=${arch} \
				--criterion=${criterion} \
				--label-smoothing=${label_smoothing} \
				--batch-size=${batch_size} \
				--update-freq=${update_freq} \
				--encoder-normalize-before \
				--decoder-normalize-before \
				--share-decoder-input-output-embed \
				--share-all-embeddings \
				--layernorm-embedding \
				--patch-layernorm-embedding \
				--code-layernorm-embedding \
				--resnet-drop-path-rate=${resnet_drop_path_rate} \
				--encoder-drop-path-rate=${encoder_drop_path_rate} \
				--decoder-drop-path-rate=${decoder_drop_path_rate} \
				--dropout=${dropout} \
				--attention-dropout=${attention_dropout} \
				--weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
				--lr-scheduler=polynomial_decay --lr=${lr} \
				--max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
				--log-format=simple --log-interval=10 \
				--fixed-validation-seed=7 \
				--keep-best-checkpoints=1 \
				--save-interval=1 --validate-interval=1 \
				--save-interval-updates=500 --validate-interval-updates=500 \
				--best-checkpoint-metric=loss \
				--max-src-length=${max_src_length} \
				--max-tgt-length=${max_tgt_length} \
				--find-unused-parameters \
				--add-type-embedding \
				--scale-attn \
				--scale-fc \
				--scale-heads \
				--disable-entangle \
				--num-bins=${num_bins} \
				--patch-image-size=${patch_image_size} \
				--prompt-type=${prompt_type} \
				--fp16 \
				--fp16-scale-window=512 \
				--num-workers=0 > ${log_file} 2>&1
	done
    done
done

# 	--minimize-best-checkpoint-metric \
# > ${log_file} 2>&1
# --eval-args="{\"sampling\":1, \"sampling-topp\": 0.95}"
# \
#     --eval-print-samples
   
