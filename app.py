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

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # グラフの実行
            inputs = {"question": prompt}
            # 同期環境で非同期を実行
            config = {"recursion_limit": 10}
            result = asyncio.run(st.session_state.app.ainvoke(inputs, config))
            
            response = result["generation"]
            st.markdown(response)
            
            # ソースの表示
            if result.get("documents"):
                with st.expander("参照元ドキュメント"):
                    for doc in result["documents"]:
                        st.write(f"- {doc.metadata.get('name', 'Unknown Source')}")

    st.session_state.messages.append({"role": "assistant", "content": response})