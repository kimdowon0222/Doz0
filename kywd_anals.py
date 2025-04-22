###test
import time
import pandas as pd
import os
from kiwipiepy import Kiwi
from collections import Counter
from itertools import islice
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

### 1. 키워드에 해당하는 기사 추출
# df = pd.read_parquet(f"/data/sgood_data/test/tmp_main/t_ms_a_a01_itrst01_main_20250101.parquet")
# df = df[df[1] == '네이버뉴스'].iloc
# news = []
# for t in df[5]:
#     if "제주항공" in t:
#         t = t.strip()
#         news.append(t)

df = pd.read_excel('공기청정기.xlsx')
# txt = '\n'.join(df['content'].tolist())

### 2. 기사 내 주요키워드 추출(kiwi)
### Kiwi 인스턴스 생성
kiwi = Kiwi()
kiwi.add_user_word('공기 청정기', 'NNP')
kiwi.add_user_word('공기청정기', 'NNP')
extract = kiwi.extract_words(df['content'].values, min_cnt=1, max_word_len=10, min_score=0.2, pos_score=-4.0, lm_filter=False)
for word in extract:
    print(word[0])


### 키워드 추출 함수
# def extract_keywords(text, pos_filters=['NNG', 'NNP']):
#     result = kiwi.analyze(text)[0]  # 분석 결과 첫 번째 항목 가져오기
#     words = [token.form for token in result[0]  # result[0]에서 토큰 리스트 가져오기
#              if token.tag in pos_filters and len(token.form) > 1]  # 명사 필터링 + 글자 수 제한
#     return Counter(words).most_common(10)  # 가장 많이 등장한 단어 5개 추출


def extract_noun_ngrams(text, n_range=(2, 3), pos_filters=['NNG', 'NNP']):
    result = kiwi.analyze(text)[0][0]  # 형태소 분석 결과 (token list)
    nouns = [token.form for token in result if token.tag in pos_filters]
    
    ngram_counter = Counter()
    
    for n in range(n_range[0], n_range[1] + 1):
        ngrams = zip(*[islice(nouns, i, None) for i in range(n)])
        for ngram in ngrams:
            phrase = ' '.join(ngram)
            ngram_counter[phrase] += 1
    
    return ngram_counter.most_common(10)

# print(extract_noun_ngrams(txt))

# kywd = [extract_keywords(t) for t in df['title']]

