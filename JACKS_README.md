## Experiment log

0. pip install -r requirements.txt

1. replicate the snli-ve

mkdir datasets
mkdir datasets/snli_ve_data
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/snli_ve_data/snli_ve_data.zip -O datasets/snli_ve_data/snli_ve_data.zip
cd datasets/snli_ve_data
unzip snli_ve_data.zip

# small, just for testing the format...
for split in {train,dev,test}; do head -5000 snli_ve_$split\.tsv > snli_ve_$split\_5K.tsv; done;


mkdir checkpoints
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt -O checkpoints/ofa_large.pt
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt -O checkpoints/ofa_base.pt

sudo apt-get install ffmpeg

cd run_scripts/snli_ve/