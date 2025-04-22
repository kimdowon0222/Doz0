### llm
import pandas as pd
import time
import os
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
)
model.eval()

# # Parquet 파일을 DataFrame으로 읽기
# # news = []
# # for date in range(20250101, 20250101+1):
# #     df = pd.read_parquet(f"/data/sgood_data/test/tmp_main/t_ms_a_a01_itrst01_main_{date}.parquet")
# #     df = df[df[1] == '네이버뉴스']
# #     news.append(df[5])

df = pd.read_parquet(f"/data/sgood_data/test/tmp_main/t_ms_a_a01_itrst01_main_20250101.parquet")
df = df[df[1] == '네이버뉴스'].iloc[:10]
news = []
for t in df[5]:
    t = t.strip()
    news.append(t)
    
def create_prompt_1(articles):
    prompt = """
    다음 뉴스 기사들을 종합하여, 주요 이슈 1개를 추출하여 한문장으로 요약하고, 그 이슈를 추출한 근거에 대해 2~3문장으로 요약해줘.
    아래 형식처럼 각각 "의미:", "분석:"으로 시작해줘.
    <형식>
    - 의미: [한 문장으로 공기청정기 주요 이슈 1개 추출]
    - 분석: [2~3문장으로 주요 이슈 1개를 추출한 근거]

    [뉴스 기사 리스트]
    """
    # 뉴스 기사 번호와 본문을 프롬프트 형식에 맞게 추가
    for idx, article in enumerate(articles, 1):
        prompt += f"{idx}. {article}\n\n"

    return prompt

#### 자동으로 프롬프트 생성
start_time = time.time() 
instruction = create_prompt_1(news)

messages = [
    {"role": "user", "content": f"{instruction}"}
    ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens = 250,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
end_time = time.time()
# 실행 시간 계산
execution_time = end_time - start_time

# 실행 시간 출력
print(f"실행 시간: {execution_time}초")

######################################################
# df = pd.read_parquet(f"/data/sgood_data/test/tmp_main/t_ms_a_a01_itrst01_main_20250104.parquet")
# df = df[df[1] == '네이버뉴스']
# news = []
# for t in df[5]:
#     t = t.strip()
#     news.append(t)

# for t in df[5]:
#     if "엑스포" in t:
#         t = t.strip()
#         news.append(t)

# def create_prompt_2(articles):
#     prompt = """
#     먼저 뉴스 기사들을 종합하여 키워드들을 추출한 후 크게 3가지 주제로 그룹화하고,
#     각 그룹의 주제와 함께 해당하는 키워드들을 작성해줘.
#     반드시 아래 형식을 그대로 따르고, 다른 말은 하지 말 것.
    
#     <형식>
#     1. [그룹1]: [키워드1, 키워드2, 키워드3, ...]
#     2. [그룹2]: [키워드1, 키워드2, 키워드3, ...]
#     3. [그룹3]: [키워드1, 키워드2, 키워드3, ...]
    
#     [뉴스 기사 리스트]
#     """
#     # 뉴스 기사 번호와 본문을 프롬프트 형식에 맞게 추가
#     for idx, article in enumerate(articles, 1):
#         prompt += f"{idx}. {article}\n\n"
#     return prompt

# with torch.no_grad():
#     print("----------")
#     start_time = time.time() 
#     instruction = create_prompt_2(news)
#     messages = [
#         {"role": "user", "content": f"{instruction}"}
#         ]

#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     ).to(model.device)

#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = model.generate(
#         input_ids,
#         max_new_tokens = 250,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9
#     )
#     print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
#     end_time = time.time()
#     # 실행 시간 계산
#     execution_time = end_time - start_time
#     # 실행 시간 출력
#     print(f"실행 시간: {execution_time}초")

