# train_sft.py
# GPT2 기반 언어모델을 Supervised Fine-Tuning (SFT) 방식으로 학습시키는 스크립트
# 4bit quantization, PEFT(LoRA) 등의 경량화 옵션을 지원

# conda activate hk_sft

import sys
import logging

# 필수 라이브러리
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,     # Causal Language Model (GPT 계열) 로딩용
    BitsAndBytesConfig,       # 4bit quantization 설정
    HfArgumentParser,         # CLI 인자 파서
    TrainingArguments,        # 학습 설정
    AutoTokenizer,            # 토크나이저 로딩
    Trainer                   # HuggingFace 기본 Trainer
)

# LoRA 관련 모듈 (Parameter-Efficient Fine-Tuning)
from peft import (
    LoraConfig,               # LoRA 설정
    PeftModel,
    get_peft_model,          # 기존 모델에 LoRA 적용
    prepare_model_for_kbit_training  # 4bit 학습용 준비 함수
)

# 현재는 TRL의 SFTTrainer는 사용하지 않음. 대신 HuggingFace Trainer 사용 (Trainer)
from trl import SFTTrainer

# 사용자 정의 모듈들
from dataset_sft import build_dataset_SFT, MyDataCollatorWithPadding, SFTDataset
from config_sft import ModelArguments, DatasetArgs, TrainArguments

import wandb 
from transformers import TrainerCallback, TrainerState, TrainerControl

# wandb 콜백 설정
class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs, step=state.global_step)

# 로깅 설정
logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_explicit_format()

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 명령줄 인자 파싱 (데이터셋, 학습, 모델 관련)
parser = HfArgumentParser((DatasetArgs, TrainArguments, ModelArguments))
data_args, training_args, model_args = parser.parse_args_into_dataclasses()

# 학습 및 검증용 데이터셋 생성
train_dataset, eval_dataset = build_dataset_SFT(data_args)

# 토크나이저 로드 및 설정
tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
collator = MyDataCollatorWithPadding(tokenizer, padding="longest", return_tensors="np")
tokenizer.pad_token = tokenizer.eos_token # eos_token을 padding token으로 설정

# eval_dataset이 비어 있는데 평가를 하도록 설정된 경우 에러 처리
if not eval_dataset and training_args.do_eval:
    raise ValueError("No eval dataset found for do_eval. Indicate eval_dataset_size argument.")


# ===== 모델 로딩 =====

# (선택) 4bit 모델 로딩
if model_args.load_in_4bit:
    if device == "cuda":
        # 4bit 양자화 설정: nf4 타입, bfloat16 연산 사용
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # 4bit 모델 로딩
        model = AutoModelForCausalLM.from_pretrained(
            model_args.pretrained_model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_auth_token=model_args.use_auth_token,
            quantization_config=quantization_config,
        )
        # 4bit 학습 준비 (gradient checkpoint 등 설정)
        model = prepare_model_for_kbit_training(model)
        logger.info(f"Training in 4 bits on")
    else:
        raise ValueError("Cannot load model in 4 bits. No cuda detected!")
# 일반 모델 로딩 (양자화 안하는 경우)
else:  
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
    )

# ===== LoRA 적용 =====
if model_args.use_peft:
    peft_config = LoraConfig(
        r=16,                    # Rank of the low-rank decomposition
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.05,       # Dropout 적용
        bias="none",             # bias는 학습하지 않음
        task_type="CAUSAL_LM"    # GPT 계열은 Causal LM
    )
    # 기존 모델에 LoRA adapter 추가
    model = get_peft_model(model, peft_config)
    logger.info(f"Training with LoRA on")

logger.info(f"Training/evaluation parameters {training_args}")

# === wandb init ===
wandb.init(
    project="SFT_GPT2",
    entity="hails",
    name=f"sft_run_{model_args.pretrained_model_name_or_path.split('/')[-1]}",
    config={**vars(data_args), **vars(training_args), **vars(model_args)}
)

# ===== Trainer 설정 =====
trainer = Trainer(
    model=model,  # 학습할 모델
    args=TrainingArguments(**vars(training_args)),  # 학습 설정 전달
    train_dataset=SFTDataset(train_dataset, tokenizer, data_args.max_seq_len),  # 학습 데이터셋
    eval_dataset=SFTDataset(eval_dataset, tokenizer, data_args.max_seq_len) if training_args.do_eval else None,
    data_collator=collator,  # 배치 패딩 처리
    tokenizer=tokenizer,       # 로그 등에서 사용됨
    callbacks=[WandbLoggingCallback()] # wandb 콜백 추가
)

trainer.train()