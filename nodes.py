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
    print("---RETRIEVING---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # 先ほど作成した vectorstore_r1 を読み込む
    vectorstore = FAISS.load_local("vectorstore_r1", embeddings, allow_dangerous_deserialization=True)
    
    documents = vectorstore.similarity_search(state["question"], k=4)
    return {"documents": documents, "question": state["question"]}

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
    if not state["documents"]:
        return "transform_query"
    return "generate"

async def transform_query(state):
    print("---TRANSFORMING QUERY---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = "より良い検索ができるように、質問を書き換えてください。"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    
    rewriter = prompt | llm | StrOutputParser()
    better_question = rewriter.invoke({"question": state["question"]})
    return {"question": better_question}