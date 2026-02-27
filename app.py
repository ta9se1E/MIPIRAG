import os
os.environ["USER_AGENT"] = "MIPIRAG/1.0"

import streamlit as st
import asyncio
from dotenv import load_dotenv
from graph_engine import compile_workflow

# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"
# .envファイルを読み込む
load_dotenv(dotenv_path=".env")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-book"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(page_title="Adaptive RAG (Vector Only)", layout="wide")
st.title("📚 研究論文 RAG システム")

# ワークフローの初期化
if "app" not in st.session_state:
    st.session_state.app = compile_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("MIやPIに関する質問をお願いします。"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # app.py (生成部分の表示を強化)

    with st.chat_message("assistant"):
        with st.spinner("思考中... 日英両方の文献をスキャンしています..."):
            inputs = {"question": prompt}
            result = asyncio.run(st.session_state.app.ainvoke(inputs))
            
            response = result["generation"]
            st.markdown(response)
            
            # --- 引用元の原文表示セクション ---
            if result.get("documents"):
                st.markdown("---")
                st.subheader("📍 引用元および参照箇所（原文）")
                
                for i, doc in enumerate(result["documents"]):
                    source_name = doc.metadata.get('name', '不明なファイル')
                    page_num = doc.metadata.get('page', '不明') # make_vector側でpageを保存している場合
                    
                    with st.expander(f"出典 {i+1}: {source_name} (Page: {page_num})"):
                        # 言語判定を簡易的に行い、タグを表示
                        lang_tag = "🇺🇸 English" if any(ord(c) < 128 for c in doc.page_content[:100]) else "🇯🇵 Japanese"
                        st.caption(f"Language: {lang_tag}")
                        st.info(doc.page_content) # ここにPDFから抽出された原文が表示される

    st.session_state.messages.append({"role": "assistant", "content": response})