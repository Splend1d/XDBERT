# The name of experiment
txt_model=bert-base-uncased
name=${txt_model}_clip_x2_wiki103_mlm_whole_seq_fast_lr5e-5linearramp
if [ ${txt_model} == "electra-large-discriminator" ]
then
    txt_model=google/${txt_model}
fi
# Create dirs and make backup
output=./results/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/xattn_pretrain.py \
    --model_name_or_path ${txt_model} \
    --config_name ${txt_model} \
    --tokenizer_name bert-base-uncased \
    --taskMatched \
    --taskMaskLM \
    --train train --valid val \
    --xlayers 2 \
    --batchSize 8 --optim bert --lr 5e-5 --epochs 1 \
    --tqdm --output $output ${@:2}  \
    --max_seq_len 512 \
    --gradAccumulation 64 \
    --multiGPU \
    --freezePretrained 0 \
    #--load results/pretrain/bert-base-uncased_clip_x2_wiki_mlm_vlm_match_whole_seq2/Epoch20/Epoch20 \
    #--overwrite_cache \
    #--gpus $1 \
    #--taskMatched  \

    #--overwrite_cache \

# for 3090, TITANRTX : bs = 8, grad acc = 64