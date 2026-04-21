# Schema for structured output
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# .env 파일에서 환경변수(ex: GOOGLE_API_KEY)를 읽어옵니다.
load_dotenv()

# Google Gemini API Key가 정상적으로 로드되었는지 검증
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("❌ .env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")

# Gemini 모델 객체 생성 (기본적으로 빠르고 우수한 'gemini-2.5-flash'를 세팅함)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Ollama 모델 객체 생성 (기본적으로 빠르고 우수한 'llama3.2:1b'를 세팅함)
ollama_llm = ChatOllama(model="llama3.2:1b")

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )
query = input("어떤모델을 사용하시겠습니까? (1: Gemini, 2: Ollama): ")
if query == "1":
    llm = gemini_llm
    print(f"✅ Gemini 모델 객체가 성공적으로 생성되었습니다: {llm.model}")
else:
    llm = ollama_llm
    print(f"✅ Ollama 모델 객체가 성공적으로 생성되었습니다: {llm.model}")

# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
print(output)
# =================================================================
# SearchQuery 모델을 활용한 Agent의 완성형 Structured Output 기법
# =================================================================
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import json

# 1. 정보 탐색을 위한 일반 도구
@tool
def normal_web_search(keyword: str) -> str:
    """웹 검색을 수행하여 정보를 탐색합니다."""
    print(f"\n🔍 [인터넷 검색 중...] {keyword}")
    return f"'{keyword}' 검색완료: 콜레스테롤 수치가 높으면 관상동맥에 칼슘 침착이 증가할 수 있다는 연구가 있습니다."

# 2. 구조화된 최종 답변 반환용 특수 도구 (SearchQuery 스키마 강제)
@tool("final_answer_submit_tool", args_schema=SearchQuery)
def final_answer_submit_tool(search_query: str, justification: str) -> str:
    """모든 조사가 끝나면, 에이전트는 반드시 이 도구를 호출하여 최종 검색어와 선택 이유를 제출하고 작업을 종료해야 합니다."""
    print("\n✅ [최종 응답 제출 도구 실행됨]")
    # 도구가 실행했다는 신호만 반환함 (중요한 건 AI가 맞춰넣은 args 인자값)
    return "SUCCESS"

tools = [normal_web_search, final_answer_submit_tool]

# 3. 에이전트 생성 프롬프트에 '최종 구조화 반환' 룰 확립
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "당신은 리서치 전문가입니다. "
        "필요시 'normal_web_search'를 사용하여 배경 지식을 먼저 검색하세요. "
        "생각을 마쳤다면, 제일 마지막에는 무조건 'final_answer_submit_tool' 도구를 사용하여 "
        "사용자 요청에 부합하는 최적의 검색어(search_query)와 그 이유(justification)를 분리해서 제출하고 종료해야 합니다."
    )
)

print("\n===========================================")
print("🤖 Agent Structured Output (Response Tool 기법)")
print("===========================================")

test_input = "높은 콜레스테롤과 심장 칼슘 CT 점수는 어떤 연관이 있어?"
initial_state = {"messages": [HumanMessage(content=test_input)]}

# 4. 에이전트 실행 루프
state_response = agent.invoke(initial_state)

print("\n✨ 전체 메시지 흐름 및 AI 자율 판단 로그:")
for msg in state_response["messages"]:
    sender = msg.__class__.__name__
    
    # 툴 콜(함수 호출)이 있는 메시지는 툴의 인자값(Args)을 보여줌
    if getattr(msg, 'tool_calls', None):
        content = f"[도구 호출 판단 🤔] {msg.tool_calls}"
    else:
        content = msg.content if msg.content else "[도구 실행 결과 반환됨]"
        
    print(f"[{sender}] {content}")

# 5. 최종 핵심 (에이전트로부터 구조화된 JSON 데이터만 쓱 빼내기)
extracted_structured_data = None
for msg in reversed(state_response["messages"]): # 통상 마지막 AI 메시지에 답이 있음
    if getattr(msg, 'tool_calls', None):
        for tc in msg.tool_calls:
            if tc["name"] == "final_answer_submit_tool":
                extracted_structured_data = tc["args"]
                break
        if extracted_structured_data:
            break

print("\n🎯 획득한 완벽한 딕셔너리(Structured) 데이터:")
print(json.dumps(extracted_structured_data, ensure_ascii=False, indent=2))