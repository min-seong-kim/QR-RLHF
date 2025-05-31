# generate_sft.py
# conda activate hk_sft

import torch
from transformers import (
    AutoTokenizer,                 # 사전 학습된 토크나이저 로드
    AutoModelForCausalLM,          # GPT2와 같은 Causal LM 모델 로드
    StoppingCriteria,              # 커스텀 stopping 조건 정의 클래스
    StoppingCriteriaList,          # 여러 조건 묶음
    HfArgumentParser               # CLI 인자 파싱
)

from config_sft import ModelArguments, GenerationArguments # 사용자 정의 인자 클래스 (argparse처럼 동작)


device = "cuda" if torch.cuda.is_available() else "cpu"

# CLI로부터 model과 generation 관련 인자를 파싱 (예: --pretrained_model_name_or_path, --max_new_tokens 등)
parser = HfArgumentParser((ModelArguments, GenerationArguments))
model_args, generation_args = parser.parse_args_into_dataclasses()

# ===== Stopping Criteria 정의 (사용자 지정 종료 조건) =====
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        # 종료를 감지할 stop 토큰 시퀀스들 (cuda로 이동)
        self.stops = [stop.to("cuda") for stop in stops]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 현재 생성된 input_ids의 마지막 토큰이 stop 조건과 일치하는지 확인
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True # 조건에 해당하면 생성 중단
        return False # 계속 생성

# ===== Stop 단어 -> 토크나이즈 -> 조건 객체로 변환 =====
def stopping_criteria(tokenizer, stop_words):
    # stop_words를 토크나이즈하여 stop token ID 시퀀스로 변환
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    
    # StoppingCriteriaList에 커스텀 조건 넣기
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria

# ===== 모델 및 토크나이저 로드 =====
tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_args.pretrained_model_name_or_path,
    device_map="auto"  # GPU/CPU 자동 배치 (multi-GPU도 가능)
)

# "\n\nHuman:"이 출력에 다시 나타나면 생성을 멈추도록 설정
stopping = stopping_criteria(tokenizer, ["\n\nHuman:"])

# ===== 입력 프롬프트 구성 및 토크나이즈 =====
inputs = tokenizer(
        "\n\nHuman: Hi, can you give me ideas on what to do during the weekend?\n\nAssistant:", 
        return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()} # 입력 텐서를 device로 이동

# ===== 텍스트 생성 실행 =====
out = model.generate(
    **inputs,                       # 입력 텐서
    stopping_criteria=stopping,    # 커스텀 종료 조건
    **vars(generation_args)        # max_new_tokens 등 추가 인자
)

# ===== 결과 디코딩 및 출력 =====
print(tokenizer.batch_decode(out))  # 생성된 토큰 → 문자열
