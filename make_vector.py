#make_vector.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/2.0"

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
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredMarkdownLoader,TextLoader 
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

# 設定
MARKDOWN_INPUT_DIR = "./input"
FAISS_SAVE_PATH = "./vectorstore_r2"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

def create_or_update_vectorstore(MARKDOWN_INPUT_DIR, FAISS_SAVE_PATH , EMBEDDINGS):
    # 1. 新規作成または読み込み
    if os.path.exists(os.path.join(FAISS_SAVE_PATH, "index.faiss")):
        print("📁 既存のFAISSインデックスを読み込みます...")
        vectorstore = FAISS.load_local(FAISS_SAVE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    else:
        print("🆕 新規作成を開始します...")
        vectorstore = None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []
    
    # フォルダ構造を考慮した探索
    for root, dirs, files in os.walk(MARKDOWN_INPUT_DIR):
        for file in files:
            if file.endswith(".md"): 
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                print(f"📄 処理中: {file_path} (論文: {folder_name})")
                
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    # 分割処理を追加
                    split_docs = text_splitter.split_documents(docs)
                    
                    # メタデータ付与
                    for doc in split_docs:
                        doc.metadata["source_paper"] = folder_name
                    
                    # リストに追加！
                    all_docs.extend(split_docs)
                except Exception as e:
                    print(f"❌ エラー ({file_path}): {e}")
    
    # 3. 追加処理
    if all_docs:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(all_docs, EMBEDDINGS)
        else:
            # すでに学習済みのファイルとの重複を避ける工夫が必要
            vectorstore.add_documents(all_docs)
        
        vectorstore.save_local(FAISS_SAVE_PATH)
        print(f"\n✅ 保存完了: {FAISS_SAVE_PATH}")
    
    return vectorstore

if __name__ == "__main__":
    create_or_update_vectorstore(MARKDOWN_INPUT_DIR, FAISS_SAVE_PATH , EMBEDDINGS)