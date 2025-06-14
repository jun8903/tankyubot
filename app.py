import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1MiIs_URQ4Td_h9YRwN1M8SYu_NY2ZiansEfizmb8JRY/edit#gid=0'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data(ttl=43200)
def load_data():
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_url(SPREADSHEET_URL)
    worksheet = spreadsheet.sheet1
    df = get_as_dataframe(worksheet, evaluate_formulas=True)
    df = df[["タイトル", "内容", "タグ"]].dropna(subset=["タイトル", "内容"], how="any")
    df["content_combined"] = (
        df["タイトル"].astype(str) + " " +
        df["内容"].astype(str) + " " +
        df["タグ"].astype(str)
    )
    embeddings = model.encode(df["content_combined"].tolist(), convert_to_tensor=True)
    return df, embeddings

df, embeddings = load_data()

# タイトル
st.title("探究チャットボット（チャットUI版）")

# チャット入力
user_input = st.chat_input("キーワードを入力してください")

if user_input:
    # ユーザーの入力を表示
    with st.chat_message("user"):
        st.write(user_input)

    # ベクトル化と類似度計算
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=3)

    # アシスタントメッセージとして表示
    with st.chat_message("assistant"):
        for i, idx in enumerate(top_results.indices):
            idx_int = idx.item()
            st.markdown(f"**順位 {i+1}**")
            st.markdown(f"**タイトル**: {df.iloc[idx_int]['タイトル']}")
            st.markdown(f"**内容**: {df.iloc[idx_int]['内容']}")
            st.markdown(f"**タグ**: {df.iloc[idx_int]['タグ']}")
            st.markdown(f"**類似度スコア**: {scores[idx_int].item():.3f}")
            st.markdown("---")
