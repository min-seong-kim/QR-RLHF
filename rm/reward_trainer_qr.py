import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from accelerate import Accelerator
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers.deepspeed import is_deepspeed_zero3_enabled

import time
import os 
import logging
import time, math, json
from typing import Callable, Dict, Any, Union, Callable, Optional, Tuple, List
 
from utils import *
from .reward_datahelper import *
from metric import MeanMetric, PPLMetric,SumMetric,RealtimeMetric,Metrics

import wandb


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

class TrainerState:
    def __init__(self) -> None:
        self.total_steps = 0
        self.total_exps = 0
        self.best_score = -9999
    
    def state_dict(self):
        return {
            'total_steps': self.total_steps,
            'total_exps': self.total_exps,
            'best_score': self.best_score,
        }
        
    def load_state_dict(self, state_dict):
        self.total_steps = state_dict['total_steps']
        self.total_exps = state_dict['total_exps']
        self.patient = state_dict['patient']
        self.best_score = state_dict['best_score']
    

class RewardTrainer():
    def __init__(self, opt, model: nn.Module, accelerator, eval_only=False) -> None:
        self.model = model
        self.accelerator = accelerator
        self.opt = opt
        self.eval_only = eval_only
        
        
        self.calculate_lm_loss: bool = opt.reward_lm_loss_factor > 0.
        self.lm_loss_factor: float = opt.reward_lm_loss_factor
        
        self.no_reset_metric_names = ['total_exs'] # metrics won't be reset for every interval steps
        self.print_interval = opt.print_interval
            
        if not eval_only:
            self.optimizer = self.build_optimizer()
            self.scheduler = self.build_scheduler()
            self.train_loader = self.build_dataloader(mode='train')
            self.train_metrics = self.build_metrics('train')
            self.train_size = len(self.train_loader.dataset)
            
        self.tokenizer = get_tokenizer(opt)
        self.valid_metrics = self.build_metrics('valid')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')
        
        self.train_state = TrainerState()
        self.max_steps: int = opt.train_steps
        self.save_per_step = opt.save_per_step
        self.model_save_path = opt.model_save_path
        self.validation_metric = opt.validation_metric
        self.fp32_loss: bool = opt.fp32_loss
        assert self.validation_metric in getattr(self, 'train_metrics', self.valid_metrics).metrics, '--validation_metric is not specified in metrics'
        
        # prepare all things, DO **NOT** prepare dataloader
        self.accelerator.register_for_checkpointing(self.train_state)
        if not self.eval_only:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
                
        synchronize_if_distributed()
        
        
        
    def build_metrics(self, mode='train'):
        assert mode in ('train', 'valid')
        metrics = Metrics(self.opt, mode=mode, accelerator=self.accelerator)
        metrics.create_metric('rm_acc', MeanMetric())
        metrics.create_metric('rm_acc_margin', MeanMetric())
        metrics.create_metric('rm_loss', MeanMetric())
        metrics.create_metric('clen', MeanMetric())
        metrics.create_metric('ctrunc', MeanMetric())
        metrics.create_metric('total_exs', SumMetric())
        
        
        if self.calculate_lm_loss:
            metrics.create_metric('ppl',PPLMetric())
            metrics.create_metric('token_acc',MeanMetric())
            metrics.create_metric('lm_loss', MeanMetric())
            
        if mode =='train':
            metrics.create_metric('lr',RealtimeMetric())
            metrics.create_metric('tpb', MeanMetric())
            metrics.create_metric('expb', MeanMetric())
            metrics.create_metric('ups', MeanMetric())
        return metrics
    
    def _run_fake_forward(self):
        # a dummy forward for stage3 compatibility to avoid deadlock
        self.model(decoder_input=torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long, device=self.accelerator.device))
    
    
    def _lm_loss(self, scores:torch.Tensor, preds:torch.Tensor, labels:torch.LongTensor, training=True):
        # calculate loss
        score_view = scores.reshape(-1, scores.size(-1)) # bs * num_tokens, vocab_size
        loss = self.loss_fn(score_view if not self.fp32_loss else score_view.to(torch.float32), labels.reshape(-1)).sum()
        if self.fp32_loss:
            loss = loss.to(scores.dtype) # cast back
        
        # calculate token acc
        notnull = labels.ne(self.tokenizer.pad_token_id)
        target_tokens = notnull.sum()
        correct = ((labels == preds) * notnull).sum()
        
        # average losses
        loss = loss / target_tokens

        # logs
        with torch.no_grad():
            obj_metrics = self._get_metric_obj(training)
            obj_metrics.record_metric('rm_loss', loss.item())
            obj_metrics.record_metric('ppl', loss.item())
            obj_metrics.record_metric('token_acc', (correct / target_tokens).item())

        return loss
    
    # def _criterion(self, model_output: torch.Tensor, batch: Dict[str, Any], return_output=False, training=True):
    #     logits, *outputs = model_output
    #     bs = logits.size(0) // 2
        
    #     preferred_rewards = logits[:bs]
    #     rejected_rewards = logits[bs:]
        
    #     probs = torch.sigmoid(preferred_rewards - rejected_rewards)
    #     loss = (-torch.log(probs + 1e-5)).mean()
                
    #     # calculate lm loss
    #     if self.calculate_lm_loss:
    #         lm_logits, *_ = outputs
    #         scores = lm_logits[:bs, :-1, :] # lm loss for chosen only
    #         preds = scores.argmax(dim=-1)
            
    #         label_vec = batch['text_vec'][:bs, 1:].clone()
    #         loss_mask = batch['loss_mask'][:, 1:]
    #         label_vec[~loss_mask] = self.tokenizer.pad_token_id
    #         batch['label_vec'] = label_vec
            
    #         lm_loss = self._lm_loss(scores, preds, label_vec, training)
    #         loss = loss + self.lm_loss_factor * lm_loss
            
    #     with torch.no_grad():
    #         correct = (probs > .5).float().mean()
    #         metrics = self._get_metric_obj(training)
    #         metrics.record_metric('rm_loss', loss.item())
    #         metrics.record_metric('rm_acc', correct.item())
            
    #     if return_output:
    #         return (loss, model_output)
    #     return loss
    
    def _quantile_huber_loss(self, preds: torch.Tensor, targets: torch.Tensor, taus: torch.Tensor):
        # preds: [B, N], targets: [B], taus: [N]
        td_errors = targets.unsqueeze(1) - preds  # [B, N]
        huber_loss = torch.where(
            td_errors.abs() <= 1.0,
            0.5 * td_errors.pow(2),
            td_errors.abs() - 0.5
        )
        weight = (taus - (td_errors.detach() < 0).float()).abs()  # [B, N]
        loss = (weight * huber_loss).sum(-1).mean()  # 평균 over quantiles and batch
        return loss

    def _criterion(self, model_output, batch, return_output=False, training=True):
        quantiles, *outputs = model_output  # quantiles: [2B, N]
        bs = quantiles.size(0) // 2  # batch size를 실제 데이터 개수로 나누기 위해 절반으로 나눔(선호 vs 거부 샘플 구성임)

        preferred_q = quantiles[:bs]  # 선호(preferred) 샘플들의 quantile 값 추출, 크기: [B, N]
        rejected_q = quantiles[bs:]   # 거부(rejected) 샘플들의 quantile 값 추출, 크기: [B, N]

        # 각 샘플(선호, 거부)별 quantile 값의 평균을 계산하여 기대값 형태로 변환
        preferred_mean = preferred_q.mean(dim=-1)  # 선호 샘플 quantile 평균값, 크기: [B]
        rejected_mean = rejected_q.mean(dim=-1)    # 거부 샘플 quantile 평균값, 크기: [B]

        # 선호 샘플이 거부 샘플보다 높은 보상을 얻는 확률을 sigmoid 함수를 통해 확률로 계산
        probs = torch.sigmoid(preferred_mean - rejected_mean)  # pairwise 승리 확률, 크기: [B]

        # 선호 샘플이 거부 샘플보다 더 높을 확률에 대한 NLL를 pairwise loss로 계산
        pairwise_loss = (-torch.log(probs + 1e-5)).mean()  # pairwise loss 계산, 수치적 안정성을 위해 1e-5를 더함

        # quantile regression에 사용할 목표값(target)을 gradients 흐름을 차단하여 설정 (안정성 향상)
        with torch.no_grad():
            target = preferred_mean.detach()  # preferred 샘플의 평균값을 타겟으로 설정함, 크기: [B]

        # quantile을 균등하게 N개의 구간으로 나누기 위해 linspace로 quantile 위치(taus)를 계산
        taus = torch.linspace(
            0 + 1 / (2 * self.model.num_quantiles),  # 첫 quantile 위치
            1 - 1 / (2 * self.model.num_quantiles),  # 마지막 quantile 위치
            self.model.num_quantiles,                # 총 quantile 개수
            device=preferred_q.device                # quantiles 텐서와 같은 디바이스에 위치
        )

        # Quantile Huber loss를 계산하여 선호 샘플의 quantile 분포가 목표값(target)을 정확히 예측하도록 유도
        quantile_loss = self._quantile_huber_loss(preferred_q, target, taus)

        # 최종 손실은 pairwise loss와 quantile loss를 합하여 계산
        loss = pairwise_loss + quantile_loss

        with torch.no_grad():
            correct = (probs > .5).float().mean()  # preferred 샘플이 높은 확률로 선택되었는지 정확도 계산
            margin_correct = (probs > 0.6).float().mean()  # 확률 변경 (0.6)로 정확도 계산
            metrics = self._get_metric_obj(training)  # 현재 모드(train/valid)에 맞는 metric 객체를 얻음
            metrics.record_metric('rm_loss', loss.item())  # 계산된 최종 loss 값을 기록
            metrics.record_metric('rm_acc', correct.item())  # 정확도(rm_acc)를 기록
            metrics.record_metric('rm_acc_margin', margin_correct.item())  

        # return_output=True일 경우, loss와 model_output(추가 정보 포함)을 함께 반환
        if return_output:
            return (loss, model_output)

        # 기본적으로 최종 계산된 손실(loss)만 반환함
        return loss

    
        
    def _save_checkpoint(self, is_best: bool, total_steps: int):
        best_model_path = os.path.join(self.model_save_path, 'best_model')
        steps_model_path = os.path.join(self.model_save_path, '{}_steps'.format(total_steps))
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if is_best:
            unwrapped_model.save_pretrained(
                best_model_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.model),
            )
            logging.info(f'Saved best model to {best_model_path}')
        
        unwrapped_model.save_pretrained(
            steps_model_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )
        logging.info(f'Saved model of {total_steps} steps to {steps_model_path}')

        synchronize_if_distributed()
    
        
    def build_dataloader(self, mode='train'):
        dataset = RMDialogDataset(self.opt, self.accelerator, mode=mode)
        return DataLoader(dataset, 
                      batch_size=None, 
                      num_workers=self.opt.num_workers, 
                      prefetch_factor=self.opt.num_prefetch, 
                      pin_memory=True, )

        
    def get_optimizer_grouped_parameters(self, model, weight_decay):
        params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n
                                for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n
                            for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return params 
    def build_optimizer(self):
        params = self.get_optimizer_grouped_parameters(self.model, self.opt.weight_decay)

        deepspeed_states = AcceleratorState().deepspeed_plugin
        if deepspeed_states.deepspeed_config['zero_optimization']['offload_optimizer']['device'] in ('none', None):
            return optim.AdamW(params, lr=self.opt.lr, eps=self.opt.eps, betas=(self.opt.beta1, self.opt.beta2))
        return DeepSpeedCPUAdam(params, lr=self.opt.lr, eps=self.opt.eps, betas=(self.opt.beta1, self.opt.beta2))

    def invsqrt_scheduler(self, warmup_steps):
        def _invsqrt_lr(step):
            return math.sqrt(warmup_steps) / math.sqrt(max(warmup_steps, step))
        def _warmup_lr(step):
            return max(step / warmup_steps, 0.1)
        def _invsqrt_lr_with_warmup(step):
            return max(_warmup_lr(step) if step < warmup_steps else _invsqrt_lr(step), 1e-8)
        
        return _invsqrt_lr_with_warmup

    def build_scheduler(self):        
        scheduler = optim.lr_scheduler.LambdaLR(
                                optimizer=self.optimizer, 
                                lr_lambda=self.invsqrt_scheduler(self.opt.warmup_steps)
                                )
        return scheduler
    
    def _get_metric_obj(self, training=True):
        if training:
            return self.train_metrics
        return self.valid_metrics
    
    @torch.no_grad()
    def _record_batch_info(self, batch, mode='train'):
        batchsize = batch.get('n_exps', batch['text_vec'].size(0))
        if mode == 'train':
            self.train_metrics.record_metric_many('clen', batch['text_len'])
            self.train_metrics.record_metric_many('ctrunc', batch['text_trunc'])
            self.train_metrics.record_metric('tpb', batch['n_tokens'])
            self.train_metrics.record_metric('expb', batchsize)
            self.train_metrics.record_metric('total_exs', batchsize)
            self.train_state.total_exps += batchsize
        elif mode == 'valid':
            self.valid_metrics.record_metric_many('clen', batch['text_len'])
            self.valid_metrics.record_metric_many('ctrunc', batch['text_trunc'])
            self.valid_metrics.record_metric('total_exs', batchsize)
        else:
            raise ValueError
    
   
    def _run_forward(self, batch: Dict[str, Any], **kwargs):
        # print('before', batch['text_vec'].dtype)
        return self.model(decoder_input=batch['text_vec'], **kwargs)
    
    def _run_fake_forward(self):
        # a dummy forward for stage3 compatibility to avoid deadlock
        self.model(decoder_input=torch.tensor([[0]], dtype=torch.long, device=self.accelerator.device))      
        
    def _train_step(self, batch: Dict[str, Any], **kwargs):
        self.optimizer.zero_grad()
        # run forward
        assert self.model.training
        model_output = self._run_forward(batch, **kwargs)
        
        # calculate loss
        loss = self._criterion(model_output, batch)
        self.accelerator.backward(loss)
        
        if torch.isnan(loss) or torch.isinf(loss) or loss.abs().gt(10000.):
            logging.warn(f'strange loss {loss.item()} detected')

        self.optimizer.step()
        if not getattr(self, 'schedule_on_valid', False) and not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
                    
    
    def _on_stop_train(self):
        return  self.train_state.total_steps >= self.max_steps
    
           
    def train(self):
        eval_score, _ = self.evaluate()
        self.train_state.best_score = eval_score
        
        synchronize_if_distributed()
        print_rank_0('Start training')
        self.model.train()
        
        while not self._on_stop_train():
            for batch in self.train_loader:
                if self._on_stop_train():
                    break
                
                start_time = time.time()
                
                self._record_batch_info(batch, mode='train')
                to_cuda(batch)
                self._train_step(batch)
                del batch
                
                cost_time = time.time() - start_time
                
                self.train_metrics.record_metric('ups', 1. / cost_time)
                if hasattr(self.scheduler, 'get_last_lr'):
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]['lr']
                self.train_metrics.record_metric('lr', lr)
                self.train_state.total_steps += 1
                
                # print metrics for every 50 steps
                need_reset = False
                if self.train_state.total_steps % self.print_interval == 0:
                    metrics = self.train_metrics.all_gather_metrics()
                    self.train_metrics.write_tensorboard(self.train_state.total_steps, gathered_metrics=metrics)
                    self.train_metrics.display(self.train_state.total_steps, self.train_size, gathered_metrics=metrics)

                    # wandb 로깅
                    wandb.log({
                        'train/rm_loss': metrics.get('rm_loss', 0),
                        'train/rm_acc': metrics.get('rm_acc', 0),
                        # 'train/ppl': metrics.get('ppl', 0),
                        'train/lm_loss': metrics.get('lm_loss', 0),
                        # 'train/token_acc': metrics.get('token_acc', 0),
                        'train/lr': metrics.get('lr', 0),
                        'train/ups': metrics.get('ups', 0),
                        # 'train/clen': metrics.get('clen', 0),
                        # 'train/ctrunc': metrics.get('ctrunc', 0),
                        'train/total_exs': metrics.get('total_exs', 0),
                    }, step=self.train_state.total_steps)

                    need_reset = True
                    
                # do evaluation for every save_per_step steps
                if self.train_state.total_steps % self.save_per_step == 0:
                    eval_score, _ = self.evaluate()
                    self.model.train()
                    
                    if any(kwd in self.validation_metric for kwd in ('loss', 'ppl')):
                        # if smaller is better
                        eval_score = -eval_score
                        
                    # save checkpoint
                    is_best = eval_score > self.train_state.best_score
                    if is_best:
                        self.train_state.best_score = eval_score
                        print_rank_0(f'Achieved the best score {abs(eval_score)}')
                    else:
                        print_rank_0(f'Did not beat the best score {abs(self.train_state.best_score)}.')
                        
                    self._save_checkpoint(is_best=is_best, total_steps=self.train_state.total_steps)
        
                if need_reset:
                    self.train_metrics.reset(no_reset=self.no_reset_metric_names)
                    
    
    @torch.no_grad()   
    def evaluate(self, datatype='valid', save_quantile=True, **kwargs) -> Tuple[float, List]:
        assert datatype in ('valid', 'test')
        start_time = time.time()
        
        valid_dataloader = self.build_dataloader(mode=datatype)
        print_rank_0(f'Start evaluation on {datatype} data.')
        self.model.eval()

        all_preferred, all_rejected, all_probs = [], [], []
        
        for step, batch in enumerate(valid_dataloader):
            # record some info
            self._record_batch_info(batch, mode='valid')
            to_cuda(batch)
            # run a forward                    
            model_output = self._run_forward(batch, **kwargs)
            # calculate and record non-generation metrics like loss and ppl

            if isinstance(model_output, tuple) and model_output[0].dim() == 2:
                quantiles = model_output[0]  # [2B, N]
                bs = quantiles.size(0) // 2
                preferred_q = quantiles[:bs].detach().to(torch.float32).cpu().numpy()
                rejected_q = quantiles[bs:].detach().to(torch.float32).cpu().numpy()
                preferred_mean = preferred_q.mean(axis=1)
                rejected_mean = rejected_q.mean(axis=1)
                probs = torch.sigmoid(torch.tensor(preferred_mean - rejected_mean)).numpy()

                all_preferred.append(preferred_q)
                all_rejected.append(rejected_q)
                all_probs.append(probs)

            self._criterion(model_output, batch, training=False)
            if is_deepspeed_zero3_enabled():
                synchronize_forward_on_stage3(False, self._run_fake_forward)
    
        # CDF용 데이터 저장
        preferred_full = np.concatenate(all_preferred, axis=0)  # [B, N]
        rejected_full = np.concatenate(all_rejected, axis=0)    # [B, N]
        # probs_full = np.concatenate(all_probs, axis=0)          # [B]

        step_str = str(self.train_state.total_steps)
        save_dir = 'outputs/csv'

        pd.DataFrame(preferred_full).to_csv(f"{save_dir}/preferred_rewards_quantile_step{step_str}.csv", index=False)
        pd.DataFrame(rejected_full).to_csv(f"{save_dir}/rejected_rewards_quantile_step{step_str}.csv", index=False)
        #pd.DataFrame({"preference_prob": probs_full}).to_csv(f"{save_dir}/preference_probs_step{step_str}.csv", index=False)

        
        if is_deepspeed_zero3_enabled():
            synchronize_forward_on_stage3(True, self._run_fake_forward)
        # log info
        metrics = self.valid_metrics.all_gather_metrics()
        self.valid_metrics.display(self.train_state.total_steps, gathered_metrics=metrics)
        self.valid_metrics.write_tensorboard(self.train_state.total_steps, gathered_metrics=metrics)
        self.valid_metrics.flush()
        validation_score = metrics[self.validation_metric]

        wandb.log({
            'eval/rm_loss': metrics.get('rm_loss', 0),
            'eval/rm_acc': metrics.get('rm_acc', 0),
            # 'eval/rm_acc_margin': metrics.get('rm_acc_margin', 0),
            # 'eval/ppl': metrics.get('ppl', 0),
            'eval/lm_loss': metrics.get('lm_loss', 0),
            # 'eval/token_acc': metrics.get('token_acc', 0),
            # 'eval/clen': metrics.get('clen', 0),
            # 'eval/ctrunc': metrics.get('ctrunc', 0),
            'eval/total_exs': metrics.get('total_exs', 0),
        }, step=self.train_state.total_steps)

        if getattr(self, 'schedule_on_valid', False) and getattr(self, 'scheduler', None) is not None:
            self.scheduler.step(validation_score)
        self.valid_metrics.reset(no_reset=[])
        
        print_rank_0(f'Evaluation completed in {(time.time() - start_time):.2f} seconds')
        return validation_score, None
    