### NAVER LLM
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# hf_token  = "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"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
model.eval()

title = "광양중앙로타리클럽, 위기가구에 중고가전 나눔"
kw_list = ["광양중앙로타리클럽", "위기가구", "중고가전", "전라남도", "광양시", "광양중앙로타리클럽", "광양읍", "관내", "이웃", "중고가전", "실천", "이웃사랑", "28일", "광양중앙로타리클럽", "2015년", "회원들", "재능기부", "후원", "집수리", "봉사", "봉사", "광양", "YWCA", "급식", "활동", "지속", "광양읍", "광양읍지역사회보장협의체", "업무협약", "체결", "생활가전", "부족", "가구", "지속적", "발굴", "지원", "중고가전", "광양읍", "맞춤", "복지팀", "복지상담", "발굴", "위기가구", "대상", "진행", "수혜", "가구", "냉장고", "고장", "일상생활", "불편", "가구", "노인", "부부", "조손", "가구", "광양중앙로타리클럽", "가정", "냉장고", "전자레인지", "밥솥", "실생활", "가전제품", "맞춤형", "지원", "회장", "박기홍", "광양중앙로타리클럽", "회원들", "마음", "이웃들", "희망", "광양읍", "맞춤", "복지팀", "협력", "해소", "복지", "사각지대", "문화", "확산", "적극", "민간위원장", "광양읍", "지역", "사회", "보장", "협의체", "민간", "위원장", "지역사회", "이웃", "지속적", "활동", "실천", "광양중앙로타리클럽", "감사", "권회상", "광양읍", "정성", "이웃", "맞춤", "가전제품", "지원", "광양중앙로타리클럽", "감사", "협력", "소외", "이웃", "실질적", "도움", "최선"]
category_list = ['수납용 가구', '사무/교구용가구', '책상 및 테이블', '의자']

start_time = time.time()
chat = [
{"role": "tool_list", "content": ""},
{"role": "system", "content": "너는 뉴스 기사를 분류하는 AI야."},
{"role": "user", "content": f"""
제목: {', '.join(title)}
키워드: {', '.join(kw_list)}
위 제목과 키워드를 보고 다음 항목 중 관련 있는 카테고리를 최대 2개까지만 선택해줘(없으면 False로 대답해줘.):
카테고리: {', '.join(category_list)}
"""}
]

with torch.no_grad():
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=False, return_dict=True, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens= 250, stop_strings=["<|endofturn|>", "<|stop|>"], tokenizer=tokenizer)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]  # 입력 길이만큼 슬라이스
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    end_time = time.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    # 실행 시간 출력
    print(f"실행 시간: {execution_time}초")

