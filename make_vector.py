import os
os.environ["USER_AGENT"] = "MIPIRAG/1.0"

import asyncio
from typing import List, Annotated, Literal, Sequence, TypedDict

import streamlit as st
from dotenv import load_dotenv

# Pydantic (v2推奨)
from pydantic import BaseModel, Field

# LangChain Core & Models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

# LangChain Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# LangChain Community (Loaders, Vectorstores, Tools)
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
from langgraph.graph import StateGraph, START, END

# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"
# .envファイルを読み込む
load_dotenv(dotenv_path=".env")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-book"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def make_vector():
    
    dirname = "input"
    files = []

    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path):
            files.append({"name": filename, "path": full_path})
            
    #embeddings= OpenAIEmbeddings(model="text-embedding-3-small")
    #dummy_text, dummy_id = "1", 1
    #vectorstore = FAISS.from_texts([dummy_text], embeddings, ids=[dummy_id])
    #vectorstore.delete([dummy_id])
    
    return files



def create_vectorstore(files):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = None

    # 1. さらに小さめのサイズで分割（1000文字 ≒ 約1500〜2000トークン程度）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "、", " ", ""] # 最後に空文字を入れて強制切断を有効化
    )

    for file in files:
        print(f"📄 処理中: {file['name']}")
        try:
            loader = PyPDFLoader(file["path"])
            pages = loader.load()
            if not pages: continue

            for page in pages:
                page.metadata["source"] = file["path"]
                page.metadata["name"] = file["name"]

            # 2. 分割実行
            raw_docs = text_splitter.split_documents(pages)
            
            # 3. 【重要】物理的な文字列スライスによる二重の安全策
            final_docs = []
            for doc in raw_docs:
                content = doc.page_content
                # 3000文字(絶対安全圏)を超えていたら、強制的に切り刻む
                if len(content) > 3000:
                    for i in range(0, len(content), 2000):
                        final_docs.append(Document(
                            page_content=content[i:i+2000], 
                            metadata=doc.metadata
                        ))
                else:
                    final_docs.append(doc)

            # 4. 【重要】1件ずつベクトルストアに追加する（一括送信による制限を回避）
            if final_docs:
                for i in range(0, len(final_docs), 10): # 10件ずつの小バッチで処理
                    batch = final_docs[i:i+10]
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        vectorstore.add_documents(batch)
                print(f"   -> {len(final_docs)} 個のチャンクを処理完了")

        except Exception as e:
            print(f"❌ エラー発生 ({file['name']}): {e}")
            continue

    if vectorstore:
        vectorstore.save_local("./vectorstore_r1")
        print("\n✅ 保存完了！")
    return vectorstore

if __name__ == "__main__":
    files = make_vector() # 前回のファイルリスト取得関数
    if files:
        create_vectorstore(files)