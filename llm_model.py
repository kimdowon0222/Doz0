###test
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)
# 모델을 여러 GPU에 나눠서 올림
# model.to(device)
model.eval()

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
instruction = """
다음 뉴스 기사들을 종합하여, 주요 이슈 1개를 추출하여 한문장으로 요약하고, 그 이슈를 추출한 근거에 대해 2~3문장으로 요약해줘. 각각 "의미:", "분석:"으로 시작해줘.

<형식>
- 의미: [한 문장으로 공기청정기 주요 이슈 1개 추출]
- 분석: [2~3문장으로 주요 이슈 1개를 추출한 근거]

[뉴스 기사 리스트]
1. 
서울=연합뉴스) 한지은 기자 = LG전자[066570]는 공감지능(AI)으로 실내 공기 질을 관리하는 '퓨리케어 오브제컬렉션 AI+ 360˚ 공기청정기'를 21일 출시한다고 밝혔다.
신제품에는 LG전자가 개발한 'AI 공기질 센서'가 처음 탑재됐다. AI 공기질 센서는 AI로 오염원을 감지하고 가스 종류와 오염도에 따라 공기 청정을 한다.
한국표준협회 테스트에서 딥러닝으로 학습한 AI 공기질 센서는 폼알데하이드, 암모니아, 휘발성유기화합물(TVOCs) 등 유해가스 3종과 유증기를 감지하고 공기를 관리하는 성능을 검증받았다.
신제품은 기존 미세먼지, 초미세먼지, 극초미세먼지, 유해가스인 휘발성유기화합물 등을 감지하는 센서와 함께 AI 공기질 센서로 총 9종의 오염원을 감지한다.
'AI 맞춤 운전 기능'으로 매시간 실내 공기 질을 학습·분석하는 것도 강점이다.
이 기능은 제품이 알아서 공기 질을 분석하고 동작 세기를 조절해 기존 AI 모드 대비 소비 전력을 최대 50% 이상 줄일 수 있다.
LG전자는 AI 공기질 센서와 AI 맞춤 운전 성능을 검증받아 한국표준협회와 국가 공인 시험인증기관 와이즈스톤이 증명하는 '에이아이플러스(AI+) 인증'을 획득했다.
국내에서 공기질 센서로 AI+ 인증을 받은 것은 이번이 처음이라고 회사는 설명했다.
신제품은 또 차세대 필터인 '퓨리탈취청정 M필터'를 적용해 기존 '퓨리탈취청정 G필터' 대비 탈취 성능이 40% 이상 향상됐다.
AI 공기질 센서가 분석한 데이터를 기반으로 '펫 특화 필터', '새집 특화 필터', '유증기 특화 필터' 등 효과적인 공기 질 관리법도 LG 씽큐 앱을 통해 추천한다.
홍순열 LG전자 ES사업본부 에어케어사업담당은 "공감지능을 강화한 혁신적인 공기 질 관리 설루션으로 실내 공기 청정과 위생 등 차별화된 고객경험을 제공할 것"이라고 말했다.

2.
(서울=연합뉴스) 장하나 기자 = 삼성전자[005930]는 6일 인공지능(AI) 기능을 기반으로 한 프리미엄 공기청정기 '비스포크 큐브 에어 인피니트 라인'을 출시한다고 밝혔다.
이미지 확대삼성전자, '비스포크 큐브 에어 인피니트 라인' 출시
삼성전자, '비스포크 큐브 에어 인피니트 라인' 출시
[삼성전자 제공. 재판매 및 DB 금지]
프리미엄 가전 라인업인 비스포크 큐브 에어 인피니트 라인은 '4방향(way) 서라운드 청정' 기술을 적용해 4면 360도 방향으로 오염된 공기를 흡입하고 깨끗해진 공기를 공간 전체에 고르게 내보낸다.
빠른 청정과 공기 순환이 필요할 때는 제품 상단에 위치한 '팝업 청정 부스터'가 작동해 필터를 통과한 청정한 공기를 최대 11m의 먼 곳까지 보낸다.
이 부스터는 스마트싱스 앱을 통해 회전 각도 범위를 설정해 주로 생활하는 공간을 맞춤 케어하는 것도 가능하며, 작동하지 않을 때는 내부에 숨겨지도록 디자인됐다.
극세필터, 항균 집진필터, 숯 탈취 강화필터로 구성된 일체형 'S필터'가 적용돼 초미세먼지 기준인 2.5마이크로미터(㎛)보다 작은 0.01㎛ 크기의 먼지를 99.999% 제거하고, 생활 냄새부터 펫 냄새까지 최대 99% 제거해준다.
AI 기술을 적용한 통합 맞춤 청정 솔루션도 특징이다.
실내외 공기질을 비교·학습해 공기질이 나빠질 것으로 예측되면 미리 실내 공기를 정화하는 '맞춤청정 AI+', 실내 공기질이 좋아지면 알아서 바람 세기를 조절하거나 팬 작동을 멈춰 에너지를 100㎡ 모델 기준 최대 45% 절감하는 'AI 절약모드' 등이 적용됐다.
'맞춤청정 AI+'는 한국표준협회에서 국제표준을 기반으로 인증하는 'AI+인증'을 받았다.
이 제품은 최근 독일 국제 디자인 공모전 'iF 디자인 어워드 2024'의 제품 부문에서 수상하기도 했다.
이무형 삼성전자 DA사업부 부사장은 "프리미엄 공기청정기에 대한 소비자의 기대를 반영해 기술과 디자인 모두 한층 업그레이드했다"며 "앞으로도 소비자가 집안에서 보내는 시간과 공간에 '변함 없는 가치'를 제공하는 차별화된 솔루션을 개발할 것"이라고 말했다.
삼성전자는 올해 AI 가전 시대를 맞아 AI 기능이 강화된 제품을 대거 선보여 'AI 가전=삼성'이라는 공식을 확고히 한다는 계획이다.
한편, 삼성전자는 오는 8~25일 비스포크 큐브 에어 인피니트 라인을 직접 체험하고 사용 경험을 공유할 앰버서더를 모집한다.

3.
중소·중견기업이 생산한 일부 공기청정기가 유해가스 제거 능력이나 소음기준 등을 충족하지 못한 것으로 드러났다.
한국소비자원은 한국과 중국 중소·중견기업이 생산한 10만∼20만원대의 공기청정기 8개 제품 성능을 평가한 결과 4개 제품은 유해가스 제거·탈취효율이 기준에 미달했고 2개 제품은 소음 기준을 충족하지 못했다고 14일 밝혔다. 4개 제품은 새집증후군 유발 물질인 폼알데하이드와 톨루엔, 대표적인 생활악취인 암모니아와 아세트알데하이드, 초산 등 5개 가스 제거율이 기준에 못 미친 것으로 나타났다.
최대 바람량으로 공기청정기를 운전했을 때 발생하는 소음을 측정한 결과 에어웰99(HK1705)와 한솔일렉트로닉스(HAP-1318A1) 등 2개 제품이 50데시벨을 초과해 관련 기준에 부적합했다. 또 소비자원이 각 제품의 필터를 확인해보니 씽크웨이 제품(ThinkAir AD24S) 필터에선 사용금지 유해성분(CMIT·MIT)이 검출됐다. 해당 제품 유통사는 유해 성분이 검출된 필터를 폐기하고, 이미 판매된 제품에 대해서는 필터를 무상으로 교체해주기로 했다.
8개 공기청정기 제품의 연간 에너지 비용은 최대 4배, 필터 교체 비용은 최대 10배 넘게 각각 차이가 났다. 공기청정기를 최대 바람량으로 하루 7.2시간씩 1년간 틀었을 때 전기요금은 8000원∼3만2000원까지 벌어졌다. 필터는 제품별로 권장 교체 주기는 6개월∼12개월로 차이가 있고 비용은 연간 1만5000원에서 18만4800원으로 격차가 컸다. 다만 8개 제품은 공기청정기 작동 시 집진에 의한 미세먼지 제거성능을 면적으로 환산한 값인 '표준사용 면적' 기준을 모두 충족했고, 구조·전기적 안전성과 오존 발생량도 모두 안전기준에 적합한 것으로 확인됐다.
소비자원은 이번 시험평가 결과를 '소비자24' 사이트에 공개하는 한편 품질 등이 미흡한 제품 제조·판매업체에 개선을 권고하기로 했다. 소비자원 관계자는 "품질이 상대적으로 우수한 것으로 평가된 브랜드는 향후 공기청정기 품질비교시험 대상에 포함시켜 소비자 선택권을 확대해 나갈 예정"이라고 전했다.

4.
청호나이스가 특화된 공기 청정 기술을 통해 겨울철 실내 공기질 개선 방법을 제안했다.
27일 업계에 따르면 청호나이스는 실내 공기질 개선을 위해 ‘항균 공기청정기 디오’와 '뉴히어로 2'를 선보였다.
신제품 항균 공기청정기 디오는 ‘스마트 AI모드’를 통해 실내 공기의 오염도를 실시간 모니터링하고 자동으로 공기질을 개선한다. AI절전모드는 공기가 깨끗한 상태로 지속될 경우 팬 작동을 멈춰 에너지 소비를 줄여주며, 공기질 매우나쁨 단계가 3분 이상 지속될 경우 AI 쾌속모드로 공기를 빠르게 정화한다.
항균 공기청정기 디오는 4단계 필터 청정 시스템인 ▲프리 필터 ▲기능성 미디엄 필터 ▲항균 집진 필터 ▲탈취 특화 필터로 구성됐다. 탈취 특화 필터는 고성능 활성탄 적용으로 기존 필터 대비 탈취 능력이 크게 향상됐으며 생활악취와 반려동물 분뇨 등 냄새 제거에 효과적이다. 
청호나이스 '뉴히어로 2'는 공기 흐름을 이상적으로 제어하는 공기역학적 설계를 지니고 있으며, 원통형 구조로 360° 전 방향에서 미세먼지를 흡입할 수 있다. 또한 세 방향(상·좌·우)으로 강력하고 빠르게 깨끗한 바람을 만들어준다.
특히 바닥에서 약 10㎝ 띄워져 있는 하부흡입 기능을 통해 바닥에서 생활하는 아이들 공간 케어에 효과적이며 바닥에 가라앉은 먼지 입자 제거에도 효과적이다. 자동(AUTO) 모드 선택 시 오염도에 따라 자동으로 청정 강도를 조절해 준다.
최근 공기청정에 관심이 있는 가구가 크게 늘며 청호나이스의 올해 10~11월 공기청정기 판매량은 전년 동기 대비 20% 증가했다.
청호나이스 관계자는 “노약자와 어린이와 같은 건강 취약계층은 미세먼지로 가득한 실내 공기에 취약하기 때문에 자사의 공기청정기를 통해 실내 공기질을 청정하고 쾌적하게 유지하시길 바란다”고 말했다.

5.
[안동=뉴시스] 류상현 기자 = 지난 해 학교 교실에 보급된 공기청정기의 임대료가 대폭 오른 이유에 대한 경북교육청의 해명이 '사실을 오도하고 있다'는 주장이 나왔다.
국민의힘 이주환 국회의원은 최근 "지난해 경북·광주·인천교육청의 공기청정기 입찰가격이 2019년 계약 때보다 1만원 이상 올랐다"며 "담합이 의심되며 이 지역 모두 한 회사 제품이 공급됐다"고 의혹을 제기했다.
이에 대해 경북교육청은 "교육부의 강화된 공기청정기 사양 기준(소음기준 55dB 이하→50dB 이하) 적용에 따라 신형기기를 도입하다 보니 예산 증가는 자연스러운 현상"이라고 언론에 설명하고 있다.
또 "일부에서 지난해 사용된 공기청정기가 구형이라는 말들이 있는데 이는 잘못된 정보이고 신형 기종을 도입해 교실 내 소음문제를 해소했다"며 "공기청정기 성능 사항을 2019년보다 2022년 개선을 하다 보니까 예산이 증액된 부분은 있었지만 담합 의혹 등은 교육청의 권한 밖의 사안"이라고 해명하고 있다.
이에 대해 지난해 뉴시스의 최초 제보자는 12일 "경북교육청의 주장은 모두 사실과 다르다"며 "지난해 납품된 공기청정기는 이미 2019년 9월 이전에 현재의 기준에 맞는 인증서를 취득했다. 기준이 강화돼 제품 가격이 높아졌다는 주장은 사실과 맞지 않다"고 주장했다.
또 "설사 기준을 강화해 부품 값이 올랐다고 해도 극히 미미한 수준이며 임대료 인상에는 전혀 영향을 미치치 않는다"고 말했다.
이와 함께 "지난해 납품된 공기청정기는 2019년에 납품된 것과 차이가 하나도 없으며 같은 회사 제품"이라며 "2019년에 2만원도 안 하던 임대료가 지난해는 4만원이 넘게 됐다는 것은 담합이 아니고서는 불가능하다"고 설명했다.
그리고 "기계값 인상으로 임대료가 높아졌다는 것이 바로 담합 업체들의 주장"이라며 "작년에 내가 경북교육청에 담합 사실을 알렸을 때 입찰을 중지했더라면 지금도 진행되고 있는 엄청난 국고손실을 막을 수 있었는데 아무도 관심이 없었다. 명백한 직무유기였다"고 주장했다.
한편 경북경찰청은 공기청정기의 가격을 담합한 혐의로 임대업체 관계자와 경북교육청 당시 업무 담당자 등 여러 명을 불구속 입건한 것으로 알려지고 있다.
뉴시스는 지난해 4월 5일, 7일 이 제보자의 제보를 바탕으로 '경북교육청, 입찰 담합 의혹에 소극 대처…업체 주장 파문'(4월 5일), '경북도내 멀쩡한 교실 공기청정기 수만대 폐기…전국적으로는?'(4월 7일) 등 교실 공기청정기 관련 내용을 보도한 바 있다.

"""

with torch.no_grad():
    start_time = time.time()
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
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
        max_new_tokens=250,
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