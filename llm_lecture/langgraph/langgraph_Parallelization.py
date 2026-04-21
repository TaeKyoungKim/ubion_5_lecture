from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# .env 파일에서 환경변수 로드
load_dotenv()

# Google Gemini API Key 확인
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("❌ .env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")

# Gemini 모델 객체 생성 (기본적으로 빠르고 우수한 'gemini-2.5-flash'를 세팅함)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Ollama 모델 객체 생성 (기본적으로 빠르고 우수한 'llama3.2:1b'를 세팅함)
ollama_llm = ChatOllama(model="llama3.2:1b")

select = input("어떤모델을 사용하시겠습니까? (1: Gemini, 2: Ollama): ")
if select == "1":
    llm = gemini_llm
    print(f"✅ Gemini 모델 객체가 성공적으로 생성되었습니다: {llm.model}")
else:
    llm = ollama_llm
    print(f"✅ Ollama 모델 객체가 성공적으로 생성되었습니다: {llm.model}")


# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# Nodes
def call_llm_1(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: State):
    """Combine the joke, story and poem into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow (파이썬 스크립트 실행 환경을 고려해 파일로 저장 추가)
try:
    display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
except NameError:
    pass

try:
    png_bytes = parallel_workflow.get_graph().draw_mermaid_png()
    with open("langgraph_workflow_parallelization.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 워크플로우 다이어그램이 'langgraph_workflow_parallelization.png' 파일로 저장되었습니다.")
except Exception as e:
    print(f"⚠️ 그래프 이미지 저장 실패: {e}")

# Invoke
print("\n🚀 병렬로 농담, 스토리, 시를 생성 중입니다... 🚀\n")
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])

