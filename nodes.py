#node.py
import os
os.environ["USER_AGENT"] = "MIPIRAG/2.0"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
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

# --- スコアリング用のデータ構造 ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevant: 'yes' or 'no'")

# --- Retrieverの設定 ---
# グローバル変数として保持し、初回のみロードするように最適化
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # 新しく作成した vectorstore_r2 を読み込む
        vectorstore = FAISS.load_local("./vectorstore_r2", embeddings, allow_dangerous_deserialization=True)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return _retriever

# --- ノード関数 ---
async def retrieve(state):
    """新しいベクトルストアから関連文書を取得"""
    print("---RETRIEVING FROM VECTORSTORE_R2---")
    retriever = get_retriever()
    # questionが変換されている場合もそのまま対応可能
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
    prompt = ChatPromptTemplate.from_messages([
        ("system", "以下のコンテキストのみを使用して回答してください。\n\nContext: {context}"),
        ("human", "Question: {question}")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": generation}
