import time
import math
import random
import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
from config_ppo import parse_args
from ppo.ppo_trainer import PPOTrainer
from ppo.ppo_datahelper import get_tokenizer
from utils import *
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import wandb

# gpt 용
from transformers import AutoModelForCausalLM, AutoConfig 
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GPT2(GPT2LMHeadModel):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer

    def forward(self, decoder_input, incr_state=None):
    
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        if incr_state is not None:
            decoder_input = decoder_input[:, -1:]

        output = super().forward(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            past_key_values=incr_state,
            return_dict=True,
            use_cache=not self.training,
			      )

        logits = output.logits
        new_incr_states = output.past_key_values
        
        return logits, new_incr_states


    @torch.no_grad()
    def generate(self, batch, **kwargs):
        """
        Generate response
        """        
        maxlen_res = kwargs.pop('maxlen_res', self.opt.maxlen_res)
        temperature = kwargs.pop('temperature', self.opt.temperature)
        repetition_penalty = kwargs.pop('repetition_penalty', self.opt.repetition_penalty)
        topp = kwargs.pop('topp', self.opt.topp)

        decoder_input: torch.LongTensor = batch['text_vec'] # (bsz, ...)

        # # positional embedding 1024 제한 
        # MAX_POS_EMBED = 1024
        # if decoder_input.shape[1] > MAX_POS_EMBED - maxlen_res:
        #     decoder_input = decoder_input[:, -(MAX_POS_EMBED - maxlen_res):]

        assert decoder_input[:, -1].ne(self.tokenizer.pad_token_id).all(), 'Last token should not be a padding token (you can use left padding instead).'
            
        dev = decoder_input.device
        bsz = decoder_input.size(0)

        scores = torch.zeros((bsz,), device=dev, dtype=torch.float16)
        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        decoder_input = torch.index_select(decoder_input, 0, inds)
        init_length = decoder_input.size(1)
            
        incr_state = None
        for _token in range(maxlen_res):
            if done.all():
                break
            score, incr_state, *_ = self.forward(decoder_input, incr_state)
            score = score.half()

            # now score is bs, len, vocab_size
            score = score[:, -1, :]
                
            # calculate repetition penalty
            if repetition_penalty > 1.:
                penalty_tokens = decoder_input[:, init_length:]
                penalty_scores = torch.gather(score, dim=1, index=penalty_tokens)
                penalty_scores = torch.where(penalty_scores < 0., penalty_scores * repetition_penalty, penalty_scores / repetition_penalty)
                score = score.scatter_(dim=1, index=penalty_tokens, src=penalty_scores)

            # nucleus sampling
            score = torch.softmax(score.div(temperature), dim=-1)
            probs = top_p_logits(score, topp=topp, filter_value=0)
            tok_ids = torch.multinomial(probs, 1)[:, 0]
            hyp_ids = torch.arange(probs.size(0), device=dev)
            scores = scores + probs[hyp_ids, tok_ids].log() * ~done

            tok_ids = torch.where(done, self.tokenizer.pad_token_id, tok_ids)
            decoder_input = torch.cat((decoder_input, tok_ids.unsqueeze(-1)), dim=-1)
            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

            incr_state = self._reorder_cache(incr_state, hyp_ids)

        # get all finalized candidates for each sample
        decoder_input = decoder_input[:, init_length:]
        decoder_input = decoder_input.view(bsz, -1)
        scores = scores.view(bsz, )

        lengths = decoder_input.ne(self.tokenizer.pad_token_id).sum(dim=-1)

        length_penalty = torch.pow(lengths, 1.0)
        scores /= length_penalty

        preds_scores = []
        for i in range(bsz):
            seq: torch.LongTensor = decoder_input[i, :lengths[i, ]]
            res_scores = (float(scores[i, ]), seq.tolist())
            preds_scores.append([res_scores])

        best_preds_scores = [preds[0] for preds in preds_scores]
        return best_preds_scores, preds_scores

class GPT2RewardModel(GPT2LMHeadModel):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        self.reward_head = torch.nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, decoder_input, only_last=True):
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)

        # GPT-2에서는 .model 이 아닌 .transformer 사용
        output = self.transformer(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )

        hidden_states = output.last_hidden_state

        if only_last:
            logits = self.reward_head(hidden_states[:, -1, :]).squeeze(-1) # [B,1] 단일 스칼라
        else:
            logits = self.reward_head(hidden_states).squeeze(-1)

        return (logits,)


# class GPT2QuantileCritic(GPT2LMHeadModel):
#     def __init__(self, config, opt, tokenizer):
#         super().__init__(config)
#         self.opt = opt
#         self.tokenizer = tokenizer

#         # quantile
#         self.n_quantiles = 10

#         # hidden_size → n_quantiles
#         self.value_head = torch.nn.Linear(config.hidden_size, self.n_quantiles)

#     def forward(self, decoder_input, attention_mask=None, **kwargs):
#         attention_mask = decoder_input.ne(self.tokenizer.pad_token_id) if attention_mask is None else attention_mask

#         # last_hidden_state 받아오도록
#         outputs = self.transformer(
#             input_ids=decoder_input,
#             attention_mask=attention_mask,
#             return_dict=True,
#         )
#         hidden_states = outputs.last_hidden_state  # [B, T, H]
#         quantiles = self.value_head(hidden_states)  # [B, T, N]
#         return (quantiles,)

class GPT2QuantileRewardModel(GPT2PreTrainedModel):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        self.num_quantiles = 10

        # 수정: transformer 로 명명
        self.transformer = GPT2Model(config)
        self.quantile_head = nn.Linear(config.hidden_size, self.num_quantiles)
        self.post_init()

    def forward(self, decoder_input, only_last=True):
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)

        output = self.transformer(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )
        hidden_states = output.last_hidden_state  # [B, T, H]

        if only_last:
            last_token_index = attention_mask.sum(dim=1) - 1  # [B]
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_index]  # [B, H]
            quantiles = self.quantile_head(last_hidden)  # [B, N]
        else:
            quantiles = self.quantile_head(hidden_states)  # [B, T, N]

        return (quantiles,)

def main(opt):
    # setup accelerator
    accelerator = setup_accelerator()

    # setup deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = opt.batch_size
    deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}

    # logging config
    logging.basicConfig(
        format='%(asctime)s - ' + f'Rank: {accelerator.process_index}' + ' - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # tokenizer
    tokenizer = get_tokenizer(opt)

    # load policy model
    logging.info(f"Loading policy model from: {opt.policy_model_path}...")
    policy_model = GPT2.from_pretrained(opt.policy_model_path,  opt=opt, tokenizer=tokenizer)
    policy_model._set_gradient_checkpointing(policy_model.transformer, opt.gradient_checkpoint)

    # load critic model
    logging.info(f"Loading critic model from: {opt.critic_model_path}...")
    critic_model =GPT2RewardModel.from_pretrained(opt.critic_model_path,  opt=opt, tokenizer=tokenizer)
    critic_model._set_gradient_checkpointing(critic_model.transformer, opt.gradient_checkpoint)

    # load reference model
    logging.info(f"Loading reference model from: {opt.policy_model_path}...")
    ref_model = GPT2.from_pretrained(opt.policy_model_path,  opt=opt, tokenizer=tokenizer)
    ref_model.resize_token_embeddings(len(tokenizer))

    # load reward model
    logging.info(f"Loading reward model from: {opt.critic_model_path}...")
    reward_model = GPT2QuantileRewardModel.from_pretrained(opt.critic_model_path,  opt=opt, tokenizer=tokenizer)
    # reward_model = GPT2RewardModel.from_pretrained(opt.critic_model_path,  opt=opt, tokenizer=tokenizer)
    reward_model.resize_token_embeddings(len(tokenizer))

    synchronize_if_distributed()

    logging.info('==================Lets go to train==================')

    # train
    trainer = PPOTrainer(opt, policy_model, ref_model, critic_model, reward_model, accelerator)
    trainer.train()

    logging.info('==================Congrats! Training completed, exit process...==================')

if __name__ == '__main__':
    opt = parse_args()
    print_rank_0(opt)
    
    wandb.init(
        project="KCC_GPT2_RLHF",
        entity="hails",  # Replace with your WandB entity
        name='PPO_Train_QR',
        config=opt,
    )

    main(opt)
