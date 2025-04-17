# #!/bin/bash
# # Copyright (c) Microsoft Corporation.
# # SPDX-License-Identifier: Apache-2.0

# export CUDA_HOME=/usr/local/cuda-11.7
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
# export PATH=${CUDA_HOME}/bin:${PATH}


# CUDA_VISIBLE_DEVICES=0 \
# accelerate launch --config_file config.yaml --num_processes 8 train_rm.py \
# --hf_model_name_or_path hf-llama-7b \
# --model_save_path ./models/Llama2/Llama-2-7b-hf \
# --batch_size 4 \
# --context_truncate 2048 \
# --data_path ./data/hh-rlhf \
# --train_steps 1000 \
# --warmup_steps 100 \
# --save_per_step 500 \
# --print_interval 5 \
# --validation_metric rm_loss \
# --lr 5e-6 \
# --reward_lm_loss_factor 0 \
# --gradient_checkpoint \
# --delimiter '</s>' \
# --logdir ./tensorboard_log/rm \
# &> /home/hail/hong_kim/MOSS-RLHF/outputs/log/training_reward.log

# # --reward_lm_loss_factor 控制在训练RM是是否计算LM loss。 如>0，则同时计算在chosen样本上的LM loss。如==0；则仅计算RM loss。


#!/bin/bash
# GPT-2 기반 reward model 학습용 스크립트

CUDA_VISIBLE_DEVICES=0 \
accelerate launch --config_file accelerate_config.yaml train_rm.py \
--hf_model_name_or_path openai-community/gpt2 \
--model_save_path ./models/reward_model_gpt2 \
--batch_size 4 \
--data_path ./data/hh-rlhf \
--train_steps 1000 \
--warmup_steps 100 \
--save_per_step 200 \
--print_interval 10 \
--validation_metric rm_loss \
--lr 5e-6 \
--reward_lm_loss_factor 0.1 \
--gradient_checkpoint \
--logdir ./tensorboard_log/rm_gpt2 \
&> ./outputs/log/reward_model_gpt2.log
