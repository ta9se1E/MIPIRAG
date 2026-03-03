import os
os.environ["USER_AGENT"] = "MIPIRAG/3.0"
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
st.title("📚 研究論文 RAG システム (画像対応版)")

# ワークフローの初期化
if "app" not in st.session_state:
    st.session_state.app = compile_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# --- サイドバー ---
with st.sidebar:
    st.header("システム設定")
    if st.button("🚀 システムを評価する (LangSmith)"):
        with st.spinner("評価データセットを検証中..."):
            from evaluation import evaluate_rag_system
            evaluate_rag_system(st.session_state.app)
            st.success("評価が完了しました！")

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                st.image(img_path, caption="参照画像")

# --- チャット処理 ---
if prompt := st.chat_input("MIやPIに関する質問を入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("思考プロセスを開始します...", expanded=True)
        
        try:
            status.write("🔍 文献・図を検索・照合中...")
            result = asyncio.run(st.session_state.app.ainvoke({"question": prompt}))
            
            response = result["generation"]
            status.update(label="✅ 回答が生成されました", state="complete")
            st.markdown(response)
            
            # --- 画像表示ロジック ---
            found_images = []
            if result.get("documents"):
                st.markdown("---")
                st.subheader("🖼️ 関連する図")
                
                # 画像タイプのドキュメントのみ抽出して表示
                image_docs = [d for d in result["documents"] if d.metadata.get("type") == "image"]
                
                if image_docs:
                    for doc in image_docs:
                        img_path = doc.metadata.get("image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"出典: {doc.metadata.get('source_paper')}")
                            found_images.append(img_path)
                else:
                    st.info("関連する図は見つかりませんでした。")

                # --- 引用元の原文セクション ---
                st.subheader("📍 引用元テキスト")
                for i, doc in enumerate(result["documents"]):
                    if doc.metadata.get("type") == "text":
                        source = doc.metadata.get('source_paper', '不明な論文')
                        with st.expander(f"出典 {i+1}: {source}"):
                            st.markdown(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": response, "images": found_images})

        except Exception as e:
            status.update(label="❌ エラーが発生しました", state="error")
            st.error(f"エラー内容: {str(e)}")
        