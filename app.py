import os
os.environ["USER_AGENT"] = "MIPIRAG/2.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    
# --- サイドバー (機能メニュー) ---
with st.sidebar:
    st.header("システム設定")
    if st.button("🚀 システムを評価する (LangSmith)"):
        with st.spinner("評価データセットを検証中..."):
            from evaluation import evaluate_rag_system
            evaluate_rag_system(st.session_state.app)
            st.success("評価が完了しました！LangSmithのコンソールを確認してください。")

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- チャット処理 ---
if prompt := st.chat_input("MIやPIに関する質問を入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # プロセスの可視化
        status = st.status("思考プロセスを開始します...", expanded=True)
        
        try:
            status.write("🔍 文献を検索・照合中...")
            # 非同期実行
            result = asyncio.run(st.session_state.app.ainvoke({"question": prompt}))
            
            response = result["generation"]
            status.update(label="✅ 回答が生成されました", state="complete")
            st.markdown(response)
            
            # --- 引用元の原文セクション ---
            if result.get("documents"):
                st.markdown("---")
                st.subheader("📍 引用元および参照箇所")
                
                for i, doc in enumerate(result["documents"]):
                    # メタデータの取得を安全に
                    source = doc.metadata.get('source_paper', '不明な論文')
                    
                    with st.expander(f"出典 {i+1}: {source}"):
                        lang = "🇺🇸 English" if any(ord(c) < 128 for c in doc.page_content[:50]) else "🇯🇵 Japanese"
                        st.caption(f"言語: {lang}")
                        st.markdown(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            status.update(label="❌ エラーが発生しました", state="error")
            st.error(f"エラー内容: {str(e)}")
        