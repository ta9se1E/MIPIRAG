#Google colabでGPU使って回した方が早く処理が終わります。

#↓Google colab用のライブラリーインストール
# 1. ライブラリインストール
#!pip install marker-pdf[all]
# 2. Google Drive マウント（ここに論文PDFを置く）
#from google.colab import drive
#drive.mount('/content/drive')

#ライブラリー インストール
import os
from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered


# 設定
INPUT_DIR = "/content/drive/MyDrive/MI_PI_研究/参考文献"#PDFファイル保管名
OUTPUT_ROOT = "/content/drive/MyDrive/MI_PI_研究/markdown_output"#markdown出力先

# モデル準備
artifact_dict = create_model_dict()
converter = PdfConverter(artifact_dict=artifact_dict)

def process_all(OUTPUT_ROOT, INPUT_DIR):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # フォルダ内のPDFを取得
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    
    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, filename)
        doc_name = os.path.splitext(filename)[0]
        final_output_dir = os.path.join(OUTPUT_ROOT, doc_name)
        
        # 既に変換済みならスキップ
        if os.path.exists(os.path.join(final_output_dir, "document.md")):
            print(f"⏩ スキップ: {filename}")
            continue
            
        print(f"📄 処理中: {filename}")
        
        # 変換と保存
        rendered = converter(pdf_path)
        full_text, _, images = text_from_rendered(rendered)
        
        os.makedirs(final_output_dir, exist_ok=True)
        with open(os.path.join(final_output_dir, "document.md"), "w", encoding="utf-8") as f:
            f.write(full_text)
            
        fig_dir = os.path.join(final_output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for img_name, img_obj in images.items():
            img_obj.save(os.path.join(fig_dir, img_name), format="PNG")
        
        print(f"✅ 完了: {doc_name}")

if __name__ == "__main__":
    process_all(OUTPUT_ROOT=OUTPUT_ROOT, INPUT_DIR=INPUT_DIR) # 前回のファイルリスト取得関数
