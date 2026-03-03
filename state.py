#state.py
from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[any] # Documentオブジェクトのリスト
    retry_count: int  # 試行回数をカウントするフィールドを追加