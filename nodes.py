import os
os.environ["USER_AGENT"] = "MIPIRAG/1.0"

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

# --- ノード関数 ---
async def retrieve(state):
    """日本語と英語の両方のクエリを用いて検索を実行"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local("vectorstore_r1", embeddings, allow_dangerous_deserialization=True)
    
    # state["question"] から日英両方のクエリを取得して検索
    # シンプルに全体を投げても text-embedding-3-small は多言語対応なので高い精度でヒットします
    documents = vectorstore.similarity_search(state["question"], k=6) 
    return {"documents": documents}

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
        score = grader.invoke({"question": state["question"], "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
            
    return {"documents": filtered_docs}

# --- 条件分岐用 ---
def decide_to_generate(state):
    print("---DECISION: ASSESSING GRADED DOCUMENTS---")
    
    # state から現在の試行回数を取得（なければ0）
    retry_count = state.get("retry_count", 0)
    filtered_docs = state.get("documents", [])
    
    # 判定ロジック
    # ドキュメントがあり、かつ検索結果が十分なら「生成」へ
    if filtered_docs:
        print("---DECISION: GENERATE---")
        return "generate"
    
    # ドキュメントがない、または5回以上ループしているなら諦めて生成へ
    if retry_count >= 5:
        print("---DECISION: GENERATE (Max retries reached)---")
        return "generate"
    
    # それ以外は再検索
    print(f"---DECISION: TRANSFORM QUERY (Attempt: {retry_count + 1})---")
    return "transform_query"


async def transform_query(state):
    """日本語の質問から、英語文献検索用のクエリも生成する"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system = """あなたは専門的なリライターです。
ユーザーの質問を、日本語での検索と英語での検索（Materials Informatics/Process Intelligence分野）の両方に最適化された形式に変換してください。
出力は以下の形式で答えてください:
Japanese Query: [日本語のクエリ]
English Query: [英語のクエリ]"""
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    rewriter = prompt | llm | StrOutputParser()
    
    full_query = rewriter.invoke({"question": state["question"]})
    # 簡単なパースで日本語と英語のクエリを抽出
    current_count = state.get("retry_count", 0)
    
    return {
        "question": full_query,
        "retry_count": current_count + 1  # ここで更新する
    }
