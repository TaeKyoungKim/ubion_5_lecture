import os
# 가장 먼저 환경 변수를 강제로 설정 (Chroma DB 내 protobuf 충돌 해결)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv

# 사용자가 지정한 기반 코드 임포트
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 기존에 작성된 utils.pdf_embbedings 파일에서 문서 파싱 함수 가져오기
from utils.pdf_embbedings import split_pdf_document

# 환경 변수 로드 (.env)
load_dotenv(".env")

def setup_and_search():
    """
    1. 문서를 로드/분할하고,
    2. 생성된 문서를 기반 코드로 ChromaDB에 저장한 뒤,
    3. 사용자 질의를 받아 검색 결과를 출력합니다.
    """
    # 임베딩 함수 초기화
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    # 1. 제시해주신 코드를 기반으로 Chroma DB 객체 초기화
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # 로컬 저장소
    )
    
    # ==========================================
    # [데이터 저장 (Save) 프로세스]
    # ==========================================
    # 테스트용 PDF 경로 지정 (존재하는 PDF에 맞춰 수정 가능)
    pdf_path = "utils/CS매뉴얼_25년_개정판.pdf"
    
    print("1. 문서를 분할하여 가져옵니다...")
    if os.path.exists(pdf_path):
        # 방금 만든 utils에서 문서를 분할해 가져옴
        docs = split_pdf_document(pdf_path, chunk_size=800, chunk_overlap=100)
        
        print("2. Chroma DB에 임베딩하여 로컬에 저장합니다...")
        
        # ⚠️ 잠재적 에러 방지: 텍스트가 비어있는 빈 청크(Empty Chunk) 필터링
        valid_docs = [doc for doc in docs if doc.page_content.strip()]
        
        print(f"총 {len(valid_docs)}개의 문서를 안전한 단일 저장 방식(1 by 1)으로 임베딩합니다...")
        
        success_count = 0
        fail_count = 0
        
        for i, doc in enumerate(valid_docs):
            try:
                # 한 번에 1개씩만 넘겨서 배열 길이 불일치 에러를 원천 차단합니다.
                vector_store.add_documents([doc])
                success_count += 1
            except Exception as e:
                fail_count += 1
                # 보통 Google 정책 필터(Safety)에 걸려 무시된 문단에서 에러가 납니다.
                print(f"   => [경고] {i+1}번째 청크는 Google API 제약(필터 등)으로 누락되었습니다. (무시하고 계속 진행)")
                
        print(f"✅ 저장이 완료되었습니다! (성공: {success_count}개, 스킵: {fail_count}개)\n")
    else:
        print(f"⚠️ 경고: {pdf_path} 파일이 없어 문서를 새로 저장하지 않고 기존 DB만 불러옵니다.\n")


    # ==========================================
    # [데이터 검색 (Search) 프로세스]
    # ==========================================
    # 사용자로부터 request(질의) 받기
    query = input("검색할 내용을 입력하세요 (엔터 시 기본값 사용): ")
    if not query.strip():
        query = "고객 불만 응대 방법"
        
    print(f"\n🔍 [질의]: {query}")
    print("관련 문서를 검색 중입니다...\n")
    
    # 벡터 저장소에서 질의와 유사도가 높은 문서를 검색
    search_results = vector_store.similarity_search(query, k=3)
    
    # 검색된 결과 출력
    print("===== 🏆 검색 결과 (Top-3) =====")
    for i, result in enumerate(search_results):
        page = result.metadata.get('page', '페이지 알 수 없음')
        print(f"[{i+1}순위 매칭 문서] (PDF 페이지: {page})")
        print(result.page_content)
        print("-" * 50)


if __name__ == "__main__":
    setup_and_search()
