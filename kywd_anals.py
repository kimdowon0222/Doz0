###test
import time
import pandas as pd
import numpy as np
import os
from kiwipiepy import Kiwi
from collections import Counter
from itertools import islice
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

### 1. 키워드에 해당하는 기사 추출
# df = pd.read_parquet(f"/data/sgood_data/test/tmp_main/t_ms_a_a01_itrst01_main_20250101.parquet")
# df = df[df[1] == '네이버뉴스'].iloc
# news = []
# for t in df[5]:
#     if "제주항공" in t:
#         t = t.strip()
#         news.append(t)

# df = pd.read_excel('공기청정기.xlsx')
# txt = '\n'.join(df['content'].tolist())

### 2. 기사 내 주요키워드 추출(kiwi)
### Kiwi 인스턴스 생성
# kiwi = Kiwi()
# kiwi.add_user_word('공기 청정기', 'NNP')
# kiwi.add_user_word('공기청정기', 'NNP')

### kiwi 코퍼스 미등록 단어 추출
# extract = kiwi.extract_words(df['content'].values, min_cnt=1, max_word_len=10, min_score=0.2, pos_score=-4.0, lm_filter=False)
# for word in extract:
#     print(word[0])


### 명사 키워드
# def extract_keywords(text, pos_filters=['NNG', 'NNP']):
#     result = kiwi.analyze(text)[0]  # 분석 결과 첫 번째 항목 가져오기
#     words = [token.form for token in result[0]  # result[0]에서 토큰 리스트 가져오기
#              if token.tag in pos_filters and len(token.form) > 1]  # 명사 필터링 + 글자 수 제한
#     return Counter(words).most_common(10)  # 가장 많이 등장한 단어 5개 추출


### n-gram 기반 키워드
# def extract_noun_ngrams(text, n_range=(2, 3), pos_filters=['NNG', 'NNP']):
#     result = kiwi.analyze(text)[0][0]  # 형태소 분석 결과 (token list)
#     nouns = [token.form for token in result if token.tag in pos_filters]
#     ngram_counter = Counter()
    
#     for n in range(n_range[0], n_range[1] + 1):
#         ngrams = zip(*[islice(nouns, i, None) for i in range(n)])
#         for ngram in ngrams:
#             phrase = ' '.join(ngram)
#             ngram_counter[phrase] += 1
#     return ngram_counter.most_common(15)

# print(extract_noun_ngrams(txt))

# kywd = [extract_keywords(t) for t in df['title']]


###### 뉴스 기사 분류 테스트(임베딩)
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# with open("news.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# title = []
# content = []
# category = []
# for i in data['data']:
#     title.append(i['doc_title'])
#     content.append(i['paragraphs'])
#     category.append(i['doc_class']['code'])
    
# news_sample = pd.DataFrame({'title' : title, 'content': content, 'category' : category})
# news_sample.to_excel('news_sample.xlsx', index = False)

news_sample = pd.read_excel('news_sample.xlsx')
dataset = news_sample['content'].iloc[0]
category_texts = {
    '경제': "금리, 환율, 소비자물가, 중앙은행, 금융시장, 주가, 무역, 수출, 수입, 부동산, 코스피, 코스닥, 통화, 예산, 경기침체, 재정, 금융위기, 투자자",
    '정치': "대통령, 총선, 국회, 법안, 여당, 야당, 정당, 청와대, 정치권, 국무총리, 장관, 선거, 입법, 헌법재판소, 내각, 대선, 의회",
    '사회': "경찰, 사건, 범죄, 교통사고, 법원, 검찰, 노동, 재난, 화재, 폭행, 체포, 실종, 구속, 재판, 형사, 민사, 시민단체, 시위, 안전사고, 학교폭력",
    'IT과학': "AI, 인공지능, 반도체, 기술 혁신, 자율주행, 클라우드, 스타트업, 로봇, 소프트웨어, 알고리즘, 양자컴퓨터, 챗봇, 빅데이터, 5G, ICT, 사이버보안, 스마트폰, IT기업, 디지털전환",
    '국제': "미국, 중국, 북한, 우크라이나, 전쟁, 외교, 정상회담, 국제사회, UN, 유럽연합, 일본, 러시아, 핵협상, 분쟁, 국제회의, 국경, 국제정세, 중동, 무역분쟁",
    '문화': "영화, 드라마, 콘서트, 예술, 전시, 배우, 감독, 문학, 축제, 미술, 공연, 연극, 소설, 문화재, 대중문화, 케이팝, 만화, 사진전, 클래식",
    '스포츠': "야구, 축구, 올림픽, 월드컵, 선수, 경기, 우승, 대표팀, 감독, 리그, 프로야구, 골프, 농구, 배구, 체육관, 메달, 스포츠클럽, 연습경기"
}

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

batch_size = 8  # 배치 크기 조절 
tokenizer = AutoTokenizer.from_pretrained("./model/klue_bert")
model = AutoModel.from_pretrained("./model/klue_bert").to(device)
dataset = news_sample['content'].iloc[1]
dataloader = DataLoader(dataset, batch_size=batch_size)
news_embedding = []
for batch in dataloader:
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():  
        outputs = model(**inputs)
    news_embedding.append(outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy())
news_embedding = np.vstack(news_embedding)

# 카테고리 유사도 측정
for cat, desc in category_texts.items():
    inputs = tokenizer(desc, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    category_embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    cosine_sim = np.mean(cosine_similarity(news_embedding, category_embedding))
    print(cat, cosine_sim)