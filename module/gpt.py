import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["openai_api_key"])

def generate_gpt_response(user_query, summaries):
    prompt = f"""
    以下は、あるキーワードに関連する探究テーマのリストです。リストに出されたものの総括をを中高生にもわかるように簡単にまとめてください。
    さらに、あるキーワードに関連して最近話題になっているニュースや、社会のトレンド、社会課題についても中高生にわかるように紹介してください。可能であれば、直近2~3年間の話題に触れ、何年にこの話題が出たのかも示してください。この時、前文と繋がるように5文程度で記載するようにしてください。
    最低でもキーワードに関連するニュース・トレンド・社会課題はどれでもいいので３つは出すようにしてください。
    箇条書きは禁止です。
    
探究テーマ一覧：
{summaries}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは中高生向けの探究学習アドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()
