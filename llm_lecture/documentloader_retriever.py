import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# 사용자 제공 코드의 Middleware (동적 모델 선택)
try:
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
except ImportError:
    pass

# ==========================================================
# 1. & 2. 문서 파싱 및 FAISS DB 저장/로드 로직
# ==========================================================
file_path = "./utils/기술적차트분석이론및방법.pdf"
faiss_db_path = "./faiss_local_db_ko" # 새로운 임베딩을 쓰므로 경로를 바꿉니다.

# 한국어 텍스트 검색에 가장 널리 쓰이는 가벼운 성능의 HuggingFace 로컬 모델
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

# 이미 만들어둔 FAISS DB 폴더가 있는지 체크
if os.path.exists(faiss_db_path):
    print("=> 📦 이미 저장된 FAISS Vector DB를 로드합니다.")
    vectorstore = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("=> 📝 PDF를 읽고 FAISS Vector DB를 새로 생성하여 저장합니다.")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 임베딩을 거쳐 FAISS DB 생성 후 디스크에 저장
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(faiss_db_path)

# ==========================================================
# 3. 검색 툴 직접 정의 (create_retriever_tool 미사용)
# ==========================================================
@tool
def pdf_search_tool(query: str) -> str:
    """이 도구는 FAISS Vector DB에서 문서를 검색합니다. 사용자 질문에 대답하기 전에 반드시 사용하세요."""
    print(f"🔍 [Tool] '{query}' 키워드로 문서를 검색합니다...")
    
    # vectorstore를 직접 사용하여 k=3 만큼 검색
    results = vectorstore.similarity_search(query, k=3)
    
    if not results:
        return "관련된 문서 내용을 찾지 못했습니다."
    
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append(f"--- 검색 결과 {i+1} ---\n{result.page_content}")
        
    return "\n\n".join(formatted_results)

tools = [pdf_search_tool]

# ==========================================================
# 4. 로컬 Llama 3 모델 선언 및 Middleware 적용
# ==========================================================
basic_model = ChatOllama(model="llama3.2:1b", temperature=0.1)
advanced_model = ChatOllama(model="llama3.2:1b", temperature=0.1)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))

# ==========================================================
# 5. 에이전트 생성 (create_agent 사용)
# ==========================================================
agent = create_agent(
    model=basic_model,
    tools=[pdf_search_tool],
    system_prompt="당신은 문서 검색 도우미 AI입니다. 반드시 'pdf_search_tool'을 사용해 논문을 검색한 뒤 답변하세요.",
    # middleware=[dynamic_model_selection] # 필요시 주석 해제하여 사용
)

# ==========================================================
# 6. 테스트 실행
# ==========================================================
if __name__ == "__main__":
    print("🚀 로컬 Llama3 RAG Agent 실행 (Vector DB 없는 BM25 검색 모델)")
    
    query = "RSI에 관해서 설명해"
    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print("\n🤖 최종 응답:")
    final_message = response["messages"][-1]
    content = final_message.content
    
    if isinstance(content, list):
        text = content[0].get("text", str(content))
    else:
        text = content
        
    print(text)
