from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 웹 검색을 위한 추가 임포트
from playwright.sync_api import sync_playwright
import urllib.parse

# 1. State 정의 (노드 간 이동하며 데이터를 저장/공유하는 상태)
# 👉 여기서 GraphState가 정의되므로, 이후 함수에서 에러가 나지 않습니다.
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
            embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask", 
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
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
    # 인터페이스 테스트 목적상 '관련 없는 문서가 있다(True)'고 판별
    return {"irrelevant_docs_present": True} 

def rewrite_query_node(state: GraphState):
    """Re-write query (Node): 구글/웹 검색을 위해 쿼리를 재작성합니다."""
    print("▶ [Node] Re-write query: 더 나은 검색을 위해 쿼리 재작성 중...")
    better_question = f"{state['question']} (웹 검색용으로 최적화됨)"
    return {"question": better_question}

# 👉 수정된 Playwright 웹 검색 로직 적용
def web_search_node(state: GraphState):
    """Web Search (Node): Playwright를 이용하여 웹 검색을 수행하고 문서를 보충합니다."""
    print("▶ [Node] Web Search: Playwright로 외부 웹 검색 수행 중...")
    
    query = state.get("question", "")
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    web_docs = []
    
    try:
        with sync_playwright() as p:
            # 1. headless=False로 변경하여 실제 브라우저 창을 띄웁니다.
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            # 검색 페이지로 이동
            page.goto(search_url, timeout=10000)
            
            # 2. 결과가 화면에 렌더링될 때까지 최대 5초간 기다립니다. (네트워크 지연 방지)
            try:
                page.wait_for_selector('.result__snippet', timeout=5000)
            except:
                print("  -> ⚠️ 요소를 찾는 데 시간이 초과되었습니다. (봇 차단이거나 클래스명 변경 의심)")
                # 화면에 어떤 내용이 떴는지 확인하기 위해 3초간 대기
                page.wait_for_timeout(3000)
            
            # 3. 텍스트 추출 시도 (만약 .result__snippet이 없다면 본문 전체의 일부라도 가져오도록 대체)
            snippets = page.locator('.result__snippet').all_inner_texts()
            
            if not snippets:
                # DuckDuckGo HTML의 다른 클래스명(.result__body)으로 재시도
                snippets = page.locator('.result__body').all_inner_texts()

            for idx, snippet in enumerate(snippets[:3]):
                web_docs.append(f"[웹 검색 문서 {idx+1}]\n{snippet.strip()}")
                
            browser.close()
            
    except Exception as e:
        print(f"  -> ⚠️ Playwright 웹 검색 중 오류 발생: {e}")
        web_docs.append("[웹 검색 실패 또는 타임아웃]")
        
    if not web_docs:
        web_docs.append("[의미 있는 웹 검색 결과를 찾을 수 없습니다]")
        
    documents = state.get("documents", []) + web_docs
    return {"documents": documents}

def answer_node(state: GraphState):
    """Answer / Generate (Node): 수집된 문서를 바탕으로 최종 답변을 만듭니다."""
    print("▶ [Node] Generate Answer: 최종 답변 생성 중...")
    return {"answer": "LLM이 문서를 기반으로 종합한 최종 답변 내용입니다."}

# 3. 단독 실행을 위한 테스트 블록
if __name__ == "__main__":
    print("--- 🚀 함수 단독 테스트 시작 ---")
    result_state = retrieve_node({"question":"LangGraph 튜토리얼"})
    print(result_state)
    # 가상의 입력 데이터(State) 만들기
    # 앞선 노드에서 이미 문서를 하나 찾았고, 검색어는 "LangGraph 튜토리얼"이라고 가정합니다.
    dummy_state = {
        "question": "LangGraph 튜토리얼",
        "documents": ["[기존 문서 1] 로컬에서 찾은 LangGraph 관련 내용입니다."]
    }
    
    # 함수 직접 호출
    result_state = web_search_node(dummy_state)
    
    print("\n--- 📝 반환된 결과 확인 ---")
    # 기존 문서와 새로 검색된 문서가 합쳐졌는지 확인
    for idx, doc in enumerate(result_state["documents"]):
        print(doc)
        print("-" * 40)