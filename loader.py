#loader.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/3.0"
from dotenv import load_dotenv

import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import json
import glob
from pathlib import Path

# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"
# .envファイルを読み込む
load_dotenv(dotenv_path=".env")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-book"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def load_all_docs(input_dir="./input"):
    """
    指定ディレクトリ内の論文フォルダを巡回し、ドキュメント（テキストと図のメタデータ）を読み込む
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []
    
    path = Path(input_dir)
    if not path.exists():
        print(f"⚠️ 指定されたディレクトリが見つかりません: {input_dir}")
        return []

    for paper_folder in path.iterdir():
        if not paper_folder.is_dir(): continue
        
        paper_name = paper_folder.name
        
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
    
    print(f"📄 全 {len(all_docs)} 個のドキュメントを読み込みました。")
    return all_docs