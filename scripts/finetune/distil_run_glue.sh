StringVal="1e-4"
StringTask="RTE MRPC STSB CoLA SST2 QQP QNLI MNLI"
#"RTE MRPC STSB CoLA SST2 QQP QNLI MNLI"
#"google/electra-large-discriminator"
pretrain_epochs="0"
seeds="997"
for task in $StringTask; do
	for seed in $seeds; do
    
		for epoch in $pretrain_epochs; do
            model_name="main/bertxclip3/Epoch03_XATTNBERT_3"
			for val in $StringVal; do 
			
				echo ${val}
				OutDir="results/finetune/distil_${model_name}/${task}/lr-"
				OutDir+="${val}"
				OutDir+="seed-${seed}"
				mkdir -p "$OutDir"
				echo ${OutDir}
				CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src python3 ./src/finetune/x_run_glue_distil.py \
				--task_name ${task} \
				--config_name "bert-base-uncased" \
				--tokenizer_name "bert-base-uncased" \
				--model_name_or_path "results/pretrain/${model_name}.pth" \
				--xlayers 2 \
				--vis_strategy token \
				--do_train \
				--do_eval \
				--do_predict \
				--learning_rate $val \
				--weight_decay 0.00 \
				--warmup_ratio 0.1 \
				--num_train_epochs 3 \
				--max_seq_length 128 \
				--output_dir $OutDir \
				--per_device_eval_batch_size 4 \
				--per_device_train_batch_size 4 \
				--gradient_accumulation_steps 8 \
				--overwrite_output \
				--evaluation_strategy epoch \
				--save_strategy no \
				--seed ${seed} \
				--save_total_limit 1\ |& tee "results/finetune/distil_${model_name}/${task}/${task}_lr${val}_seed${seed}.txt" -i
				
				#rm -r "$OutDir/*"
				#--data_dir data/glue \
				#--warmup_steps 12 \
				#--overwrite_cache \
				#--eval_steps 12 \
				#--logging_steps 12 \
				#--config_name "bert-base-cased" \
				#--tokenizer_name "bert-base-cased" \
				#--evaluate_during_training
				
				#unc-nlp/
			done
		done
	done
done
