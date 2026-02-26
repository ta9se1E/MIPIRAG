from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[any] # Documentオブジェクトのリスト