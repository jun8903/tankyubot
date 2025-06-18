import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# --- 1. モデルロード ---
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- 2. データ読み込み＆ベクトル化＆sklearnインデックス作成 ---
@st.cache_data(show_spinner=True)
def load_data_and_indexes(_model):
    df = pd.read_csv("data.csv")

    title_emb = _model.encode(df["タイトル"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    content_emb = _model.encode(df["内容"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    tag_emb = _model.encode(df["タグ"].tolist(), convert_to_tensor=True, normalize_embeddings=True)

    # sklearn用に変換
    index_title = NearestNeighbors(metric="cosine").fit(title_emb.cpu().numpy())
    index_content = NearestNeighbors(metric="cosine").fit(content_emb.cpu().numpy())
    index_tag = NearestNeighbors(metric="cosine").fit(tag_emb.cpu().numpy())

    return df, index_title, index_content, index_tag

# --- 3. 検索処理 ---
def search(df, indexes, model, user_input, k=5):
    index_title, index_content, index_tag = indexes
    query_vec = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().reshape(1, -1)

    D_title, I_title = index_title.kneighbors(query_vec, n_neighbors=k)
    D_content, I_content = index_content.kneighbors(query_vec, n_neighbors=k)
    D_tag, I_tag = index_tag.kneighbors(query_vec, n_neighbors=k)

    # スコアに変換（cosine距離 → 類似度）
    D_title = 1 - D_title
    D_content = 1 - D_content
    D_tag = 1 - D_tag

    weight_title = 1.0
    weight_content = 2.0
    weight_tag = 0.3

    score_dict = {}

    for rank in range(k):
        idx_t = I_title[0, rank]
        score_dict[idx_t] = score_dict.get(idx_t, 0) + D_title[0, rank] * weight_title

        idx_c = I_content[0, rank]
        score_dict[idx_c] = score_dict.get(idx_c, 0) + D_content[0, rank] * weight_content

        idx_g = I_tag[0, rank]
        score_dict[idx_g] = score_dict.get(idx_g, 0) + D_tag[0, rank] * weight_tag

    top_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    threshold = 0.5
    filtered = [(idx, sc) for idx, sc in top_results if sc >= threshold]

    return filtered

# --- 4. Streamlit UI ---
def main():
    st.title("探究テーマ検索システム")

    model = load_model()
    df, index_title, index_content, index_tag = load_data_and_indexes(model)

    user_input = st.text_input("キーワードを入力してください")

    if user_input:
        results = search(df, (index_title, index_content, index_tag), model, user_input, k=10)

        if not results:
            st.write("該当する探究テーマが見つかりませんでした。")
        else:
            for rank, (idx, score) in enumerate(results[:5], start=1):
                st.markdown(f"### {rank}位 (スコア: {score:.3f})")
                st.write(f"**タイトル:** {df.iloc[idx]['タイトル']}")
                st.write(f"**内容:** {df.iloc[idx]['内容']}")
                st.write(f"**タグ:** {df.iloc[idx]['タグ']}")
                st.write("---")

if __name__ == "__main__":
    main()
