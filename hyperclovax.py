### NAVER LLM
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# hf_token  = "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"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
model.eval()

start_time = time.time()
chat = [
  {"role": "tool_list", "content": ""},
  {"role": "system", "content": ""},
  {"role": "user", "content": 
  """ 
  다음은 기사에서 추출된 단어들입니다:
[
  "코웨이", "브랜드", "조사서", "입증", "시장", "신뢰", "코웨이", "브랜드", "대한민국", "브랜드스타", "대한민국", "브랜드", "스타", "한국", "산업", "브랜드파워", "K-BPI", "성과", "24일", "회사", "브랜드", "가치", "평가", "브랜드스탁", "브랜드", "대한민국", "코웨이", "연간", "순위", "대비", "상승", "32위", "브랜드", "대한민국", "특허", "BSTI", "브랜드", "가치", "평가", "모델", "Brand", "Stock", "Top", "Index", "산업", "산업", "브랜드", "점수", "브랜드", "상위", "제도", "브랜드스탁", "주관", "대한민국", "브랜드", "스타", "조사", "정수기", "정수기", "연속", "공기청정기", "비데", "3관왕", "달성", "대한민국", "브랜드", "스타", "산업", "브랜드", "가치", "선정", "인증제도", "브랜드", "가치", "평가", "인증", "제도", "평가모델", "브랜드", "가치", "평가", "모델", "BSTI", "바탕", "선정", "한국능률협회컨설팅", "KMAC", "주관", "한국", "산업", "브랜드파워", "정수기", "차지", "한국산업", "브랜드", "파워", "조사", "시작", "1999년", "27년", "1위", "차지", "가전제품", "분야", "1위", "연속", "차지", "브랜드", "코웨이", "유일", "설명", "코웨이", "조사", "정수기", "포함", "공기청정기", "비데", "1위", "연속", "차지", "환경가전제품", "1위", "달성", "코웨이", "관계자", "코웨이", "조사", "각종", "브랜드", "성과", "요인", "혁신", "제품", "아이콘", "시리즈", "노블", "시리즈", "룰루", "더블", "비데2", "시장", "주도", "출시", "혁신", "제품", "아이콘", "정수기", "아이콘", "얼음", "정수기", "아이콘", "시리즈", "크기", "세련", "디자인", "생활", "인기", "코웨이", "정수기", "160만", "누적", "판매", "돌파", "비렉스", "브랜드", "스마트", "매트리스", "안마베드", "페블체어", "트리플체어", "제품들", "소비자들", "호응", "결과", "코웨이", "판매량", "렌털", "기간", "14.1%", "증가", "171만", "달성", "서울비즈"
]

아래는 '계절가전' 카테고리의 세부 분류입니다:
- 공기청정기
- 에어컨
- 가습기
- 제습기
- 선풍기
- 전기히터
- 냉온풍기 
위 단어들을 고려할 때, 가장 관련 있는 세부 카테고리 1개 또는 2개를 선택해 주세요."""},
]

with torch.no_grad():
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=False, return_dict=True, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens= 100, stop_strings=["<|endofturn|>", "<|stop|>"], tokenizer=tokenizer)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]  # 입력 길이만큼 슬라이스
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    end_time = time.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    # 실행 시간 출력
    print(f"실행 시간: {execution_time}초")
