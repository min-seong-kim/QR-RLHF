from transformers.models.llama.modeling_llama import LlamaForCausalLM

from typing import Dict, Any, Union, List, Tuple
import os
import random
import numpy as np

from config_rm import parse_args
from utils import *
from rm.reward_trainer_qr import RewardTrainer
# from rm.reward_trainer_qr_index import RewardTrainer
from rm.reward_datahelper import get_tokenizer
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2PreTrainedModel, GPT2Model

import wandb

class GPT2QuantileRewardModel(GPT2PreTrainedModel):
    def __init__(self, config, opt, tokenizer, **kwargs):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer

        self.num_quantiles = 50  # default 값, 필요 시 opt에서 전달
        self.transformer = GPT2Model(config)

        self.quantile_head = nn.Linear(config.hidden_size, self.num_quantiles)

        # lm loss용 head (GPT2LMHeadModel에는 기본적으로 있음)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.calculate_lm_loss = getattr(opt, 'reward_lm_loss_factor', 0.) > 0.

        self.post_init()

    def forward(self, decoder_input: torch.LongTensor, rank_all=False):
        """
        Args:
            decoder_input: [B, T]
            rank_all (bool): True일 경우, 전체 토큰에 대해 quantile score를 반환
        Returns:
            - rank_all=False: [B, N]
            - rank_all=True:  [B, T, N]
        """
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)  # [B, T]

        outputs = self.transformer(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )
        hidden_states = outputs.last_hidden_state  # hidden state를 만들어 이후 quantile 예측을 위한 layer[B, T, H]

        if rank_all: # 모든 토큰 시점 [B, T]에 대해 각 시점별로 N개의 quantile 값을 반환
            quantiles = self.quantile_head(hidden_states)  # [B, T, N]
        else: # 각 시퀀스의 마지막 유효 토큰에 대해서만 quantile 값을 반환
            last_token_index = attention_mask.sum(dim=1).clamp(min=1) - 1  # [B]
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_index]  # [B, H]
            quantiles = self.quantile_head(last_hidden)  # [B, N]

        if self.calculate_lm_loss:
            lm_logits = self.lm_head(hidden_states)  # [B, T, V]
            return quantiles, lm_logits
        else:
            return (quantiles,)


def main(opt):
    # setup accelerator
    accelerator = setup_accelerator()

    # setup deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = opt.batch_size
    deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}

    # logging config
    if accelerator and accelerator.use_distributed:
        logging.basicConfig(
            format="%(asctime)s - "
            + f"Rank: {accelerator.process_index}"
            + " - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
        )
    logger = logging.getLogger(__name__)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    # load initial reward model, if init_checkpoint_model is specified, load the specified model, otherwise load the pre-trained model
    if opt.init_checkpoint_model and os.path.isdir(opt.init_checkpoint_model):
        logging.info(f"Load existing model from {opt.init_checkpoint_model}")
        model = GPT2QuantileRewardModel.from_pretrained(
            opt.init_checkpoint_model, opt, get_tokenizer(opt)
        )
    else:
        logging.info(f"Load **init** model from {opt.hf_model_name_or_path}")
        model = GPT2QuantileRewardModel.from_pretrained(
            opt.hf_model_name_or_path, opt, get_tokenizer(opt)
        )

    # set gradient checkpointing
    model._set_gradient_checkpointing(model.transformer, opt.gradient_checkpoint)

    synchronize_if_distributed()
    
    # init reward trainer and start training
    trainer = RewardTrainer(opt, model, accelerator)
    trainer.train()
    
    logging.info('==================Congrats! Training completed, exit process...==================') 

if __name__ == "__main__":
    opt = parse_args()
    print_rank_0(opt)

    wandb.init(
        project="KCC_GPT2_RLHF",
        entity="hails",  # Replace with your WandB entity
        name='Reward_Train_QR',
        config=opt,
    )
    main(opt)
