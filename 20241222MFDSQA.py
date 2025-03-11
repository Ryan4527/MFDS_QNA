# TF-IDF 적용
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

data_file = '2024MFDSQNA.csv'
log_file = os.path.join(os.getcwd(), 'Q_search_log.csv')

# 파일이 없으면 자동 생성
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["입력 문장", "입력 시간"])
    df.to_csv(log_file, index=False, encoding="utf-8-sig")

st.set_page_config(page_title="식약처 질의응답을 문장으로 검색", layout="wide")
st.title("식약처 질의응답을 문장으로 검색")
st.text("입력된 문장으로 가장 유사한 질의응답을 찾아줍니다.")
st.text("아래는 검색 예제입니다.")
st.image("example.jpg", use_column_width=True)
df = pd.read_csv(data_file)

new_post = st.text_input("검색할 질의 문장을 입력하세요.")
# new_post = ['액제 이화학적동등성 시험 선정']

def save_log(new_post, search_time):
    log_data = pd.DataFrame([[new_post, search_time]], columns=["입력 문장", "입력 시간"])

    # 기존 로그가 있으면 불러오기
    try:
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            existing_logs = pd.read_csv(log_file)
            log_data = pd.concat([existing_logs, log_data], ignore_index=True)
    except FileNotFoundError:
        pass

    log_data.to_csv(log_file, index=False, encoding="utf-8-sig")

if new_post:
    search_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    save_log(new_post, search_time)
    # st.success("log가 저장됨됨")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['QNA'].tolist() + [new_post])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similarity_scores = cosine_sim[0]
    top_indices = np.argsort(similarity_scores)[::-1][:5]
    top_scores = similarity_scores[top_indices]

    # most_similar_index = np.argmax(similarity_scores)
    # most_similar_score = similarity_scores[most_similar_index]

    # 결과 출력
    st.subheader("가장 유사한 QNA TOP 5")
    for i, index in enumerate(top_indices):
        st.write(f"**{i+1}. {df.iloc[index]['QNA']}** (유사도 점수: {top_scores[i]:.2f})")

    st.dataframe(pd.DataFrame({
        "순위": range(1, 6),
        "유사도 점수": top_scores,
        "QNA": df.iloc[top_indices]["QNA"].values
    }))


# print(f"가장 유사한 문장: {df.iloc[most_similar_index]['QNA']}")
# print(f"유사도 점수: {most_similar_score:.2f}")