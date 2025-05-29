########## 공기청정기 3개월치 이슈 추출
#### DBSCAN 모듈
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import time

def get_morphs(sen_list):
    '''
    형태소 분석기
    Args:
        sen_list: 형태소 분석할 문장 리스트
    Returns:
        morphs_list: 형태소 분석 결과 리스트
    '''    
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    morphs_list = []
    for s in tqdm(sen_list):
        tokens = kiwi.tokenize(s)
        morphs = [token.form for token in tokens]
        morphs_list.append(morphs)
    return morphs_list


class clustering_analyis():
    '''
    이슈 클러스터링 분석
    '''
    def __init__(self, emb_model_path):
        '''
        초기화 함수
        Args:
            emb_model_path (str): 임베딩 모델 경로 또는 모델 이름 (예: "home/model/klue_bert")
        '''
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
        self.model = AutoModel.from_pretrained(emb_model_path).to(self.device)

    
    def dbscan(self, df, morphs, cluster, param=0.8, eps_step=0.1, min_samples=3, ngram_range=(1,5), max_attempts=5):
        '''
        DBSCAN 군집 분석
        Args:
            morphs (str): 문자열 형태소 리스트 컬럼
            cluster(str): DBSCAN 결과 저장 컬럼
            param (float): 초기 eps 값 (default 0.8)
            eps_step(float): eps 감소단위 (default 0.1)
            min_samples (int): DBSCAN min_samples 설정 (default 3)
            ngram_range (tuple): TF-IDF n-gram 범위 (default (1,5))
            max_attempts (int): eps 감소 최대 시도 횟수 (default 5)
        Returns:
            pd.DataFrame: DBSCAN 결과가 추가된 데이터프레임
        '''
        try:
            df = df.copy()
            text = [" ".join(morph) for morph in df[morphs]]
            
            tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=ngram_range)
            tfidf_vectorizer.fit(text)
            vector = tfidf_vectorizer.transform(text)
            
            for attempt in range(max_attempts):
                eps = param - eps_step*attempt
                if eps <= 0:
                    break
                model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
                dbscan_result = model.fit_predict(vector)
                cluster_num = len(set(dbscan_result))
                if cluster_num > 1:
                    df[cluster] = dbscan_result
                    return df
        except Exception as e:
            raise

    
    def sentence_embedding(self, dataset, batch_size=8):
        '''
        문장 임베딩 및 유사도
        Args:
            dataset (list): 임베딩 대상 문자열 리스트
            batch_size(int): embedding 배치 크기 조절(default 8)
        Returns:
            embedding(vector): 임베딩 벡터
        '''    
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            embedding_vectors = []
            for batch in dataloader:
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.no_grad():  
                    outputs = self.model(**inputs)
                embedding_vectors.append(outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy())
            embedding_vectors = np.vstack(embedding_vectors)
            return embedding_vectors
        except Exception as e:
            raise
    
    def cosine_sim(self, vector_1, vector_2):   #### 코사인 유사도 input, 계산법 여러가지로
        '''
        코사인 유사도 계산
        Args:
            vector_1(vector): 벡터값
            vector_2(vector): 벡터값
        Returns:
            cosine_vector(vector): 코사인 유사도 벡터
            cosine_sim(float): 코사인 유사도 평균
        '''
        try:
            cosine_vector = cosine_similarity(vector_1, vector_2)
            cosine_sim = np.mean(cosine_similarity(vector_1, vector_2))
            return cosine_vector, cosine_sim
        except Exception as e:
            raise
    
    
    def cluster_similarity_eval(self, df, cluster, sen, similarity, batch_size = 8):
        '''
        군집 유사도 평가
        Args:
            df (pd.DataFrame): 군집 결과 데이터프레임
            cluster(str): 군집 결과 컬럼
            sen(str): 문장 텍스트 컬럼
            similarity(float): 코사인 유사도값
            batch_size(int): embedding 배치 크기 조절(default 8)
        Returns:
            pd.DataFrame: 군집 유사도 평가결과 데이터프레임
        '''
        try:
            #### 군집별 임베딩 벡터 저장
            cluster_df = df.loc[df[cluster] != -1]
            cluster_list = cluster_df[cluster].unique().tolist()
            cluster_embedding = dict()
            for cl in cluster_list:
                cluster_cl = cluster_df[cluster_df[cluster] == cl]
                vector = self.sentence_embedding(dataset = cluster_cl[sen].tolist(), batch_size=batch_size)
                cluster_embedding[cl] = vector
            
            #### 군집 내 유사도 평가
            for cl in cluster_list:
                _ ,cluster_sim = self.cosine_sim(cluster_embedding[cl], cluster_embedding[cl])
                if cluster_sim < similarity:   #### 군집 간 유사도가 낮으면 삭제
                    cluster_df = cluster_df.drop(cluster_df[cluster_df[cluster] == cl].index)
            
            #### 군집 간 유사도 평가
            while True:
                cluster_list = cluster_df[cluster].unique().tolist()
                cluster_cb = list(combinations(cluster_list,2))
                merged = False
                for cl_1, cl_2 in cluster_cb:
                    _ ,cluster_sim = self.cosine_sim(cluster_embedding[cl_1], cluster_embedding[cl_2])
                    if cluster_sim > similarity:   #### 군집 간 유사도가 높으면 통합
                        new_label = f"{cl_1}_{cl_2}"
                        cluster_df.loc[(cluster_df[cluster] == cl_1) | (cluster_df[cluster] == cl_2), cluster] = new_label
                        cluster_new = cluster_df[cluster_df[cluster] == new_label]
                        vector = self.sentence_embedding(dataset = cluster_new[sen].tolist(), batch_size=batch_size)
                        cluster_embedding[new_label] = vector
                        merged = True
                        break  
                if not merged:
                    break    
            return cluster_df
        except Exception as e:
            raise
    
    def top_k_issue(self, df, cluster, issue_rank, tnocs, topk):
        '''
        이슈 순위 선정
        Args:
            df (pd.DataFrame): 군집 결과가 포함된 데이터프레임
            cluster(str): 군집 결과 컬럼
            issue_rank(int): 이슈 순위 컬럼
            tnocs(str): 군집 데이터 개수 컬럼
            topk(int): 추출할 이슈 순위값
        Returns:
            pd.DataFrame: 군집 이슈 순위 계산결과 데이터프레임
        '''
        cluster_df = df.loc[df[cluster] != -1]
        cluster_counts_df = cluster_df[cluster].value_counts().reset_index()
        cluster_counts_df.columns = [cluster, tnocs]
        cluster_counts_df[issue_rank] = cluster_counts_df[tnocs].rank(ascending= False, method='first')
        issue_rank_df = pd.merge(cluster_df, cluster_counts_df, left_on = [cluster],
                                    right_on = [cluster], how = 'inner')
        main_issue_df = issue_rank_df[issue_rank_df[issue_rank] <= topk]
        return main_issue_df

    def cluster_main_issue(self, df, cluster, sen, issue, batch_size = 8):
        '''
        메인 이슈명 채택
        Args:
            cluster(str): DBSCAN 결과 컬럼
            sen(str): 문장 텍스트 컬럼
            issue_nm(str): 군집 이슈명 저장 컬럼
            batch_size(int): embedding 배치 크기(default 8)
        Returns:
            pd.DataFrame: 군집 이슈명 채택결과 데이터프레임
        '''
        try:
            total_issue_df= pd.DataFrame()
            cluster_df = df.loc[df[cluster] != -1]
            cluster_list = cluster_df[cluster].unique().tolist()
            
            for cl in cluster_list:
                tmp_issue_df = cluster_df[cluster_df[cluster] == cl].reset_index(drop=True).copy()
                embedding_vector = self.sentence_embedding(dataset = tmp_issue_df[sen].tolist(), batch_size=batch_size)
                cluster_sim_vec, _ = self.cosine_sim(embedding_vector, embedding_vector)
                cluster_sim_lst = [(i, np.mean(j)) for i,j in enumerate(cluster_sim_vec)]
                sim_scores = sorted(cluster_sim_lst, key=lambda x: x[1], reverse=True)
                top_idx = sim_scores[0][0]
                issue_nm = tmp_issue_df[sen].iloc[top_idx]
                tmp_issue_df[issue] = issue_nm
                total_issue_df = pd.concat([total_issue_df, tmp_issue_df])
            total_issue_df.index = range(len(total_issue_df))
            return total_issue_df
        except Exception as e:
            raise
    

df = pd.read_excel('/home/dowon/git_project/Doz0/KIDP_샘플데이터_공기청정기_202501.xlsx')
st = time.time()
title_list = df['title']
rst = get_morphs(sen_list = title_list)
df['morphs'] = rst
ca = clustering_analyis("/home/dowon/git_project/Doz0/model/klue_bert")
df = ca.dbscan(df, 'morphs', 'cluster', param=0.8, eps_step=0.1, min_samples=3, ngram_range=(1,5), max_attempts=5)
df = ca.cluster_similarity_eval(df, 'cluster', 'title', similarity= 0.8, batch_size = 8)
df = ca.top_k_issue(df, 'cluster', 'issue_rank', 'tnocs', topk= 10000)
df = ca.cluster_main_issue(df, 'cluster', 'title', 'issue', batch_size = 8)
print(time.time() - st)
df.to_excel('dbscan_KIDP_샘플데이터_공기청정기_202501.xlsx', index = False)


