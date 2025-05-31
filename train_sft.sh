#!/bin/bash

CUDA_VISIBLE_DEVICES="0, 1"
# --nproc_per_node 3 : 노드당 프로세스 3개 (GPU 3개 사용)
# pretrained_model_name_or_path: 사전 학습 모델 경로 (SFT가 이미 한 번 된 모델에서 이어서 학습)
# load_in_4bit: 모델을 4bit로 로드할지 여부 (False: full precision 사용)
# use_peft: LoRA 사용 여부 (False: full fine-tuning)

# 이거 학습 된 모델 이어서 학습시키려면 추가
# --pretrained_model_name_or_path /outputs/models/sft/gpt_tuned_sft/checkpoint-10000/ \

# conda activate hk_sft

torchrun --nproc_per_node 2 train_sft.py \
    --dataset_name Anthropic/hh-rlhf \
    --max_seq_len 750 \
    --output_dir outputs/models/sft/gpt_tuned_sft/ \
    --pretrained_model_name_or_path outputs/models/sft/gpt_tuned_sft/checkpoint-1500/ \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 500 \
    --max_steps 10000 \
    --learning_rate 3e-5 \
    --optim "adamw_hf" \
    --logging_steps 200 \
    --warmup_ratio 0.05 \
    --do_train True \
    --load_in_4bit False \
    --use_peft False \
    --optim paged_adamw_32bit \
    &> outputs/log/sft_gpt2.log
    