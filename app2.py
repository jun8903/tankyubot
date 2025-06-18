import streamlit as st
from module.model_loader import load_model
from module.data_handler import load_data_and_indexes
from module.search import search
from module.gpt import generate_gpt_response  # GPT応答

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
            summaries = ""
            for rank, (idx, score) in enumerate(results[:5], start=1):
                title = df.iloc[idx]['タイトル']
                content = df.iloc[idx]['内容']
                tags = df.iloc[idx]['タグ']

                st.markdown(f"### {rank}位 (スコア: {score:.3f})")
                st.write(f"**タイトル:** {title}")
                st.write(f"**内容:** {content}")
                st.write(f"**タグ:** {tags}")
                st.write("---")

                summaries += f"{rank}. {title} - {content}\n"

            # GPTによる全体の要約とトレンド情報
            with st.spinner("GPTによる要約とトレンドを生成中..."):
                gpt_summary = generate_gpt_response(user_input, summaries)
                st.markdown("## GPTによるまとめとトレンド解説")
                st.write(gpt_summary)

if __name__ == "__main__":
    main()
