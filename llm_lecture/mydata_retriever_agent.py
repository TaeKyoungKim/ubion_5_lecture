import os
import sys

# Windows 환경에서 백그라운드 스레드의 이모지(Emoji) 출력 시 발생하는 `cp949` 에러 방지
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 가장 먼저 환경 변수를 강제로 설정하여 Chroma DB 내 protobuf 충돌 해결
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama  # Ollama를 통한 로컬 Llama 모델 지원
from langchain.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma  # 최신 패키지 사용
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import datetime

# .env 파일을 읽어옴
load_dotenv()

# os.environ을 통해 값에 접근
google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_api_key

# 1. 사용할 LLM 모델 선택 (사용자 상호작용)
print("============================================================")
print("어떤 LLM 모델을 에이전트의 두뇌로 사용할지 선택하세요:")
print("1. Google Gemini (gemini-2.5-flash)")
print("2. Meta Llama 3.2 1B (llama3.2:1b - Ollama 로컬 구동)")
choice = input("번호를 입력하세요 (엔터 시 기본값 1번): ").strip()

if choice == "2":
    print("=> 🦙 Llama 3.2 1B 모델을 로드합니다 (속도가 조금 걸릴 수 있습니다)...\n")
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.2,
    )
else:
    print("=> ✨ Google Gemini 2.5 Flash 모델을 로드합니다...\n")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        temperature=0.2,           
        max_output_tokens=2048
    )

# 2. Vector Store 로드 (이전에 저장한 DB를 불러옴)
# run_chroma_search.py를 실행하여 DB가 생성되어 있어야 합니다.
vector_store = Chroma(
    collection_name="example_collection", # 데이터 저장 시 지정했던 이름과 동일하게 맞춰야 합니다.
    persist_directory="./chroma_langchain_db", 
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
)

# 3. Tool 정의 (LLM이 사용할 수 있는 도구 생성)
@tool
def search_my_data(query: str):
    """
    사용자의 질문(query)을 받아서, 저장된 PDF 문서에서 가장 관련성 높은 내용을 검색합니다.
    """
    print(f"🔍 [Tool] '{query}'에 대한 문서를 검색합니다...")
    
    # Vector Store를 사용하여 유사도 검색 수행 (Top 3)
    results = vector_store.similarity_search(query, k=3)
    
    # 검색된 내용을 LLM이 이해할 수 있는 텍스트로 가공
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append(f"--- 검색 결과 {i+1} ---\n{result.page_content}")
        
    # 검색된 개수를 명확히 보여주도록 디버깅 메시지를 개선했습니다.
    if formatted_results:
        print(f"👉 [Tool 디버그] 총 {len(formatted_results)}개의 관련 문서를 성공적으로 찾았습니다!")
    else:
        print(f"👉 [Tool 디버그] 관련된 문서를 찾지 못했습니다.")
        
    return "\n\n".join(formatted_results)

# 4. Agent 생성
# Agent는 도구(Tools)를 사용하여 스스로 생각하고 행동하는 AI입니다.
agent = create_agent(
    model=llm,
    tools=[search_my_data],  # 사용할 도구 리스트
    system_prompt="""
    당신은 사내 문서(PDF)를 바탕으로 사용자의 질문에 답변하는 AI 어시스턴트입니다.
    사용자가 "고객응대"와 같이 아주 짧은 단어나 모호한 키워드만 입력하더라도, 되묻지 말고 **무조건 제일 먼저 'search_my_data' 도구를 실행**하여 문서를 검색하세요.
    검색 도구를 사용하지 않고 임의로 답변하거나 역질문하는 것을 엄격히 금지합니다.
    도구를 사용해 얻은 문서 내용들을 바탕으로, 사용자가 궁금해할 만한 내용을 친절하게 요약해서 답변해주세요.
    """
)

# 5. 테스트 실행
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RAG Agent (My Data Retriever) 실행")
    print("=" * 60)
    
    while True:
        user_input = input("\n👤 사용자 질문: ")
        if user_input.lower() in ["exit", "quit", "종료"]: break
        
        # Agent 실행
        response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        
        print("\n🤖 AI 답변:")
        # response는 딕셔너리 형태의 상태(state)를 반환하므로, 마지막 메시지를 추출해야 합니다.
        final_message = response["messages"][-1]
        content = final_message.content
        
        # ChatGoogleGenerativeAI(Gemini) 최신 버전은 가끔 응답을 [{'type': 'text', 'text': '...'}] 형태의 리스트 구조로 보냅니다.
        # 이를 평문(순수 텍스트)으로 예쁘게 정제합니다.
        if isinstance(content, list):
            text = content[0].get("text", str(content))
        else:
            text = content
            
        print(text)
        print("-" * 60)