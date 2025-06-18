import pandas as pd
import faiss
import streamlit as st
import torch

@st.cache_data(show_spinner=True)
def load_data_and_indexes(_model):
    df = pd.read_csv("data.csv")

    title_emb = _model.encode(df["タイトル"].tolist(), convert_to_tensor=True)
    content_emb = _model.encode(df["内容"].tolist(), convert_to_tensor=True)
    tag_emb = _model.encode(df["タグ"].tolist(), convert_to_tensor=True)

    dim = title_emb.shape[1]

    index_title = faiss.IndexFlatIP(dim)
    index_title.add(title_emb.cpu().numpy().astype("float32"))

    index_content = faiss.IndexFlatIP(dim)
    index_content.add(content_emb.cpu().numpy().astype("float32"))

    index_tag = faiss.IndexFlatIP(dim)
    index_tag.add(tag_emb.cpu().numpy().astype("float32"))

    return df, index_title, index_content, index_tag
