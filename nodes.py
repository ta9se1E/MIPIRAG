#node.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/3.0"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever  # 0.2系での正規パス
from langchain.retrievers import ContextualCompressionRetriever # 0.2系での正規パス

# その他必要なモジュール
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from openai import OpenAI

# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"
# .envファイルを読み込む
load_dotenv(dotenv_path=".env")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API")
os.environ["LANGCHAIN_PROJECT"] = "agent-book"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# --- スコアリング用のデータ構造 ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevant: 'yes' or 'no'")

# --- Retrieverの設定 ---
# グローバル変数として保持し、初回のみロードするように最適化
_retriever = None

def get_retriever(all_docs=None):
    global _retriever
    if _retriever is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local("./vectorstore_r3", embeddings, allow_dangerous_deserialization=True)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 5
        
        # 自作のアンサンブルクラスを使用
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        
        reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
        _retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble_retriever
        )
    return _retriever

async def retrieve(state):
    print("---RETRIEVING FROM VECTORSTORE_R3 WITH HYBRID/RERANK---")
    
    # 実際には、get_retrieverを呼ぶ前に all_docs がロードされている必要があります。
    # ここでは、もし初回でなければ引数なしでも動くように工夫します
    try:
        retriever = get_retriever()
    except ValueError:
        # ここで再度ロードを試みる、あるいはエラーを投げる処理
        raise ValueError("システム起動時に Retriever が初期化されていません。")
        
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

async def grade_documents(state):
    print("---CHECKING RELEVANCE---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(GradeDocuments)
    
    system = "ドキュメントが質問に関連しているか「yes」か「no」で評価してください。"
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    grader = grade_prompt | structured_llm
    
    filtered_docs = []
    for d in state["documents"]:
        # 評価ロジックを独立させて実行
        score = grader.invoke({"question": state["question"], "document": d.page_content})
        if score.binary_score.lower() == "yes":
            filtered_docs.append(d)
            
    return {"documents": filtered_docs}

async def transform_query(state):
    """質問を検索用に最適化（英語・日本語クエリの生成）"""
    print("---TRANSFORMING QUERY---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system = """あなたは専門的なリライターです。ユーザーの質問を、日本語・英語での検索に適したクエリに変換してください。
出力は「Japanese Query: [日本語] English Query: [英語]」の形式で記述してください。"""
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    rewriter = prompt | llm | StrOutputParser()
    
    full_query = rewriter.invoke({"question": state["question"]})
    return {
        "question": full_query,
        "retry_count": state.get("retry_count", 0) + 1
    }

def decide_to_generate(state):
    """生成に進むか、クエリ変換を行うかを判断"""
    print("---ASSESSING GRADED DOCUMENTS---")
    filtered_docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    
    # 十分なドキュメントがあるか、回数制限に達したら生成へ
    if filtered_docs or retry_count >= 5:
        print("---DECISION: GENERATE---")
        return "generate"
    
    print(f"---DECISION: TRANSFORM QUERY (Attempt: {retry_count + 1})---")
    return "transform_query"

async def generate(state):
    print("---GENERATING---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # フィルタリングされたドキュメントを取得
    docs = state["documents"]
    
    # Contextを整理
    context_text = "\n\n".join([d.page_content for d in docs if d.metadata.get("type") == "text"])
    images_info = "\n\n".join([f"- 出典: {d.metadata['source_paper']}, パス: {d.metadata['image_path']}, 説明: {d.page_content}" 
                               for d in docs if d.metadata.get("type") == "image"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "以下のコンテキストを使用して回答してください。\n\n[テキスト情報]\n{context_text}\n\n[図の参照情報]\n{images_info}"),
        ("human", "Question: {question}")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({
        "context_text": context_text, 
        "images_info": images_info, 
        "question": state["question"]
    })
    
    # 【変更点】generationに加え、使用したドキュメントのメタデータを返す
    source_metadata = [{"source": d.metadata.get("source_paper"), "type": d.metadata.get("type")} for d in docs]
    
    return {
        "generation": generation,
        "documents": docs,  
        "source_metadata": source_metadata # これをグラフのstateに追加
    }
# OpenAIクライアントの初期化（get_retriever等の外で）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_diagram_node(state):
    print("---GENERATING INFOGRAPHIC---")
    rag_answer = state["generation"]
    sources = ", ".join([d.metadata.get("source_paper", "Unknown") for d in state["documents"]])
     # 1. 図解用プロンプトの生成 (LLM)
    prompt_instruction = f"""
    Based on this RAG answer: '{state["generation"]}', 
    and referencing these sources: '{sources}',
    create a detailed DALL-E 3 prompt. No text, high-quality, 16:9.
    """
    
    prompt_res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_instruction}]
    )
    final_prompt = prompt_res.choices[0].message.content
    
    # 2. 画像生成 (DALL-E 3)
    image_res = client.images.generate(
        model="dall-e-3",
        prompt=final_prompt,
        size="1792x1024",
        n=1
    )
    
    return {"image_url": image_res.data[0].url}
