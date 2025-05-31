#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# --hf_model_name_or_path openai-community/gpt2 \

export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}


CUDA_VISIBLE_DEVICES=1 \
accelerate launch --config_file accelerate_config.yaml --num_processes 1 train_rm_qr.py \
--hf_model_name_or_path outputs/models/sft/gpt_tuned_sft/checkpoint-1500/ \
--model_save_path outputs/rewardModel_qr \
--batch_size 4 \
--context_truncate 1024 \
--data_path data/hh-rlhf \
--train_steps 200000 \
--warmup_steps 20000 \
--save_per_step 2000 \
--print_interval 10 \
--validation_metric rm_loss \
--lr 2e-6 \
--reward_lm_loss_factor 0 \
--gradient_checkpoint \
--delimiter '</s>' \
--logdir outputs/tensorboard_log/reward/reward_model_qr_en \
&> outputs/log/reward_model_qr_en.log
# --reward_lm_loss_factor 控制在训练RM是是否计算LM loss。 如>0，则同时计算在chosen样本上的LM loss。如==0；则仅计算RM loss。

# --hf_model_name_or_path openai-community/gpt2 \
# --warmup_steps 10000 \