#make_vector.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/3.0"

import asyncio
from typing import List, Annotated, Literal, Sequence, TypedDict
import streamlit as st
from dotenv import load_dotenv
import json
import glob
from pathlib import Path

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
FAISS_SAVE_PATH = "./vectorstore_r3"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

def create_or_update_vectorstore(input_dir, save_path, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []
    
    # 既存のインデックスがある場合、登録済み論文名を抽出
    processed_papers = set()
    vectorstore = None
    
    if os.path.exists(os.path.join(save_path, "index.faiss")):
        print("📁 既存のインデックスを読み込みます...")
        vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        # docstoreから既存のメタデータを取得（簡易的に全てのdocのsource_paperを取得）
        processed_papers = {doc.metadata.get("source_paper") for doc in vectorstore.docstore._dict.values() if "source_paper" in doc.metadata}
        print(f"   登録済み論文数: {len(processed_papers)}")
    else:
        print("🆕 新規作成を開始します...")

    # 論文フォルダを巡回
    for paper_folder in Path(input_dir).iterdir():
        if not paper_folder.is_dir(): continue
        
        paper_name = paper_folder.name
        if paper_name in processed_papers:
            print(f"⏩ スキップ: {paper_name} (既に登録済み)")
            continue
            
        print(f"📄 処理中: {paper_name}")
        
        # 1. マークダウンの処理
        md_file = paper_folder / "document.md"
        if md_file.exists():
            loader = TextLoader(str(md_file), encoding='utf-8')
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            
            for doc in split_docs:
                doc.metadata.update({"type": "text", "source_paper": paper_name})
            all_docs.extend(split_docs)
        
        # 2. 図のメタデータ処理
        fig_meta_path = paper_folder / "figures_summary.json"
        if fig_meta_path.exists():
            with open(fig_meta_path, "r", encoding="utf-8") as f:
                figs = json.load(f)
                for fig in figs:
                    all_docs.append(Document(
                        page_content=f"図の説明: {fig['caption']}\n詳細: {fig['summary']}",
                        metadata={
                            "type": "image",
                            "source_paper": paper_name,
                            "image_path": str(Path(fig['img_path']))
                        }
                    ))
    
    # 3. ベクトルストアへの登録
    if all_docs:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(all_docs, embeddings)
        else:
            vectorstore.add_documents(all_docs)
        
        vectorstore.save_local(save_path)
        print(f"\n✅ 保存完了: {save_path}")
    else:
        print("\nℹ️ 新規登録対象のドキュメントはありませんでした。")
    
    return vectorstore

if __name__ == "__main__":
    create_or_update_vectorstore(MARKDOWN_INPUT_DIR, FAISS_SAVE_PATH, EMBEDDINGS)