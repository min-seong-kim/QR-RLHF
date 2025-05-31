#!/bin/bash
# Copyright (c) Fudan NLP Group.
# SPDX-License-Identifier: Apache-2.0

CUDA_VISIBLE_DEVICES=0  \
accelerate launch \
    --config_file accelerate_config.yaml \
train_ppo.py \
    --tokenizer_name_or_path openai-community/gpt2 \
    --policy_model_path openai-community/gpt2 \
    --critic_model_path outputs/rewardModel/10000_steps \
    --model_save_path outputs/models/ppo/ppo_model_en \
    --data_path data/ppo_data \
    --seed 42 \
    --maxlen_prompt 756 \
    --maxlen_res 256 \
    --lr 5e-7 \
    --critic_lr 1.5e-6 \
    --gamma 1. \
    --lam 0.95 \
    --entropy_clip 35.0 \
    --value_clip 0.2 \
    --pg_clip 0.2 \
    --reward_clip 0. \
    --entropy_loss_weight 0. \
    --ppo_pretrain_loss_weight 0. \
    --kl_penalty_weight 0.01 \
    --use_reward_scaling \
    --use_critic_loss_clip \
    --use_policy_loss_clip \
    --train_steps 1000 \
    --save_per_step 100 \
    --warmup_steps 100 \
    --batch_size 2 \
    --rollout_batch_size 2 \
    --num_rollouts 2 \
    --gradient_checkpoint \
    --lang en \
    --logdir outputs/tensorboard_log/ppo/ppo_model_en \
&> outputs/log/ppo_model_en.log