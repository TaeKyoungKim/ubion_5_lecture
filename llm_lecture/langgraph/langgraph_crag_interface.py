"""
Corrective RAG (CRAG) Workflow Interface
첨부하신 이미지의 CRAG(기밀성 검증이 추가된 RAG) 작동 구조를 바탕으로 구현한 
LangGraph 인터페이스 기반 코드입니다.
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
    
# 1. State 정의 (노드 간 이동하며 데이터를 저장/공유하는 상태)
class GraphState(TypedDict):
    question: str                # 사용자 질문
    documents: list              # 검색된 문서 리스트
    irrelevant_docs_present: bool # '관련 없는 문서가 있는지' (Any doc irrelevant) 여부
    answer: str                  # 최종 답변

# ───────────────────────────────────────────────
# 2. 노드(Node) 함수 정의
def retrieve_node(state: GraphState):
    """Retrieve (Node): Question을 받아 문서를 검색합니다."""
    print("▶ [Node] Retrieve: 문서 검색 중...")
    faiss_db_path = r"c:/apps/ubion_5_lecture/llm_lecture/faiss_local_db_ko"
    retrieved_docs = []
    
    if os.path.exists(faiss_db_path):
        print(f"  -> FAISS DB 연동 중... ({faiss_db_path})")
        try:
            # 한국어 임베딩 모델 (앞서 사용하신 모델 적용)
            embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask", 
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True}
            )
            # 벡터 DB 로드
            vectorstore = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
            
            # 사용자 질문에 대해 문서 검색 (Top-2)
            results = vectorstore.similarity_search(state["question"], k=2)
            
            if results:
                for idx, doc in enumerate(results):
                    retrieved_docs.append(f"[관련 문서 {idx+1}]\n{doc.page_content.strip()}")
            else:
                retrieved_docs.append("[의미론적으로 연관된 문서를 찾을 수 없습니다]")
        except Exception as e:
            print(f"  -> ⚠️ FAISS 검색 중 오류: {e}")
            retrieved_docs.append("[오류로 인한 검색 실패]")
    else:
        print(f"  -> ⚠️ FAISS DB 경로를 찾을 수 없습니다: {faiss_db_path}")
        retrieved_docs.append("[DB 경로 없음]")

    return {"documents": retrieved_docs}

def grade_node(state: GraphState):
    """Grade (Node): 수집된 문서가 질문과 관련되어 있는지 평가합니다."""
    print("▶ [Node] Grade: 문서 관련성(Relevant/Irrelevant) 평가 중...")
    # TODO: LLM 평가 로직 추가
    # 인터페이스 테스트 목적상 '관련 없는 문서가 있다(True)'고 판별
    return {"irrelevant_docs_present": True} 

def rewrite_query_node(state: GraphState):
    """Re-write query (Node): 구글/웹 검색을 위해 쿼리를 재작성합니다."""
    print("▶ [Node] Re-write query: 더 나은 검색을 위해 쿼리 재작성 중...")
    # TODO: LLM을 이용한 Query Rewrite 로직
    better_question = f"{state['question']} (웹 검색용으로 최적화됨)"
    return {"question": better_question}

def web_search_node(state: GraphState):
    """Web Search (Node): 재작성된 쿼리로 웹을 검색하여 문서를 보충합니다."""
    print("▶ [Node] Web Search: 외부 웹 검색 수행 중...")
    # TODO: Tavily, DuckDuckGo API 등 연동
    web_docs = ["[문서3] 웹에서 찾은 최신 구글/위키 정보"]
    documents = state.get("documents", []) + web_docs
    return {"documents": documents}

def answer_node(state: GraphState):
    """Answer / Generate (보라색 노드): 수집된 모든 문서를 바탕으로 최종 답변을 만듭니다."""
    print("▶ [Node] Generate Answer: 최종 답변 생성 중...")
    # TODO: 문서들을 context로 삼고 LLM에게 최종 응답 도출 지시
    return {"answer": "LLM이 문서를 기반으로 종합한 최종 답변 내용입니다."}


# ───────────────────────────────────────────────
# 3. 조건부 엣지(Conditional Edge) 로직
# ───────────────────────────────────────────────
def check_relevance(state: GraphState):
    """
    다이어그램의 다이아몬드(조건 분기) 노드:
    Any doc irrelevant (관련 없는 문서가 하나라도 있는가?)
    """
    print("\n  [Condition Check] Any doc irrelevant?")
    if state.get("irrelevant_docs_present", False):
        print("    -> Yes (무관한 문서 있음. Re-write query 로 이동합니다)")
        return "yes"
    else:
        print("    -> No (모두 유관함. Answer 노드로 바로 이동합니다)")
        return "no"


# ───────────────────────────────────────────────
# 4. 워크플로우 연결(Build Graph)
# ───────────────────────────────────────────────
def build_crag_workflow():
    workflow = StateGraph(GraphState)

    # 1) 노드(Node) 추가
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_answer", answer_node)

    # 2) 화살표(Edge) 연결
    # Question -> Retrieve
    workflow.add_edge(START, "retrieve")
    
    # Retrieve -> Grade
    workflow.add_edge("retrieve", "grade")
    
    # Grade -> Condition Box
    workflow.add_conditional_edges(
        "grade",
        check_relevance,
        {
            "yes": "rewrite_query",  # Any doc irrelevant 가 'Yes'일 때
            "no": "generate_answer"  # Any doc irrelevant 가 'No'일 때
        }
    )

    # Re-write query -> Web Search
    workflow.add_edge("rewrite_query", "web_search")
    
    # Web Search -> Answer
    workflow.add_edge("web_search", "generate_answer")

    # Answer -> 최종 결과
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ───────────────────────────────────────────────
# 실행 및 다이어그램 저장
# ───────────────────────────────────────────────
if __name__ == "__main__":
    app = build_crag_workflow()
    
    b, r = "\033[1m", "\033[0m"
    print(f"\n{b}=== LangGraph CRAG 워크플로우 인터페이스 ==={r}\n")
    
    # 워크플로우 사진(PNG) 생성
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        out_path = "langgraph_crag_diagram.png"
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ 워크플로우 구조도가 '{out_path}' 이름으로 저장되었습니다!\n")

        # 터미널 실행 시 구조도 이미지가 자동으로 열리도록 처리
        import os
        import platform
        try:
            abs_path = os.path.abspath(out_path)
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":
                import subprocess
                subprocess.call(["open", abs_path])
            else:
                import subprocess
                subprocess.call(["xdg-open", abs_path])
        except Exception as img_eval_err:
            print(f"이미지 자동 열기 실패: {img_eval_err}")

    except Exception as e:
        print(f"⚠️ 워크플로우 이미지 생성 실패 (단, Jupyter등의 그래프 의존성 에러일 수 있음): {e}\n")

    # 테스트 실행
    print(f"{b}🚀 [실행 시뮬레이션] 🚀{r}")
    initial_state = {
        "question": "LangGraph의 CRAG 패턴이 무엇인가요?", 
        "documents": [], 
        "irrelevant_docs_present": False,
        "answer": ""
    }
    
    # 그래프 실행 (흐름을 눈으로 보기 위함)
    result = app.invoke(initial_state)
    
    print(f"\n{b}=== 최종결과 ==={r}")
    print(f"- 질문: {result['question']}")
    print(f"- 문서(로컬 + 웹): {result['documents']}")
    print(f"- 답변: {result['answer']}")
    print("\n")
