#LLM_makes_json.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/3.0"

import json
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import re


# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"
# .envファイルを読み込む
load_dotenv(dotenv_path=".env")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-book"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class FigureCaption(BaseModel):
    caption: str = Field(description="画像の内容を簡潔に説明したキャプション")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption_with_llm(img_path, surrounding_text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # パーサーの設定
    parser = JsonOutputParser(pydantic_object=FigureCaption)
    
    encoded_image = encode_image(img_path)
    
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": f"""
            以下の画像の内容を要約してJSON形式で返してください。
            JSONの形式は必ず {parser.get_format_instructions()} に従ってください。
            【重要】挨拶や説明文は一切不要です。JSON文字列のみを出力してください。
            
            周囲のテキスト: {surrounding_text[:500]}
            """},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        ])
    ]
    return (llm | parser).invoke(messages)

def process_papers(base_folder):
    for paper_folder in Path(base_folder).iterdir():
        if not paper_folder.is_dir(): continue
        
        # 1. Markdownからリストを作成
        md_files = list(paper_folder.glob("*.md"))
        if not md_files: continue
        
        md_content = md_files[0].read_text(encoding="utf-8")
        matches = re.findall(r'!\[(.*?)\]\((.*?)\)', md_content)
        
        summary_data = []
        for alt_text, img_path in matches:
            # 相対パスを絶対パスのように扱うためにファイル名を抽出
            filename = Path(img_path).name
            real_img_path = paper_folder / filename
            
            summary_data.append({
                "img_path": str(real_img_path),
                "caption": alt_text if alt_text and len(alt_text) > 5 else "キャプション自動抽出不可",
                "summary": "解析前"
            })
            
        # 2. キャプションが空のものをLLMで補完
        needs_save = False
        for item in summary_data:
            # 1. 直接指定されたパス
            filename = Path(item["img_path"]).name
            possible_paths = [
                paper_folder / filename,           # 論文直下
                paper_folder / "figures" / filename # figuresフォルダ内
            ]
            
            # 2. 存在するパスを特定
            real_img_path = next((p for p in possible_paths if p.exists()), None)
            
            if item["caption"] == "キャプション自動抽出不可":
                if real_img_path:
                    print(f"--- ANALYZING: {real_img_path.name} ---")
                    try:
                        # 解析時には見つかった正しい絶対パスを使用
                        res = generate_caption_with_llm(str(real_img_path), md_content[:1000])
                        item["caption"] = res.get("caption", "要約生成失敗")
                        item["summary"] = f"図の説明: {item['caption']}"
                        item["img_path"] = str(real_img_path) # 正しいパスに更新
                        needs_save = True
                    except Exception as e:
                        print(f"❌ 解析エラー: {e}")
                else:
                    print(f"❌ ファイルが見つかりません: {filename}")
        
        # 3. JSON保存
        json_path = paper_folder / "figures_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"✅ {paper_folder.name} 完了: {len(summary_data)} 枚の図を処理")

if __name__ == "__main__":
    process_papers("./input")