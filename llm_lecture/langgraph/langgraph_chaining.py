from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
# .env 파일에서 환경변수(ex: GOOGLE_API_KEY)를 읽어옵니다.
load_dotenv()

# Google Gemini API Key가 정상적으로 로드되었는지 검증
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
    improved_joke: str
    final_joke: str


# Nodes
def generate_joke(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    print(msg.content)
    return {"joke": msg.content}


def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""

    # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"


def improve_joke(state: State):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile
chain = workflow.compile()

# Show workflow (파이썬 스크립트 실행을 위해 파일로 저장)
try:
    png_bytes = chain.get_graph().draw_mermaid_png()
    with open("langgraph_workflow.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 워크플로우 다이어그램이 'langgraph_workflow.png' 파일로 저장되었습니다.")
except Exception as e:
    print(f"⚠️ 그래프 이미지 저장 실패 (그래프 렌더링에 필요한 추가 패키지가 없을 수 있습니다): {e}")

# Invoke
state = chain.invoke({"topic": "dogs with korean"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Final joke:")
    print(state["joke"])