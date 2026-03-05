#graph_engine.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/3.0"

from langgraph.graph import StateGraph, START, END
from state import GraphState
from nodes import retrieve, generate, grade_documents, transform_query, decide_to_generate, generate_diagram_node
from dotenv import load_dotenv

# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"
# .envファイルを読み込む
load_dotenv(dotenv_path=".env")


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-book"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def compile_workflow():
    """RAGのワークフローを構築する"""
    workflow = StateGraph(GraphState)

    # 1. ノードの追加
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_diagram", generate_diagram_node) # 追加
    workflow.add_node("transform_query", transform_query)

    # 2. エッジ（処理の流れ）の定義
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # 評価に基づいた条件分岐
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    # ループ処理の接続
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "generate_diagram")
    workflow.add_edge("generate_diagram", END)
    
    # 3. コンパイル
    app = workflow.compile()
    app.recursion_limit = 10
    
    return app