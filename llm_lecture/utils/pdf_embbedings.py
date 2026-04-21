import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 환경 변수 로드 (API 키 세팅 공통 적용)
load_dotenv(".env")
api_key = os.getenv("GOOGLE_API_KEY")

def generate_sentence_embedding(text: str, model_name: str = "gemini-embedding-2-preview"):
    """
    임의의 텍스트를 받아서 벡터 임베딩을 생성하고 결과를 반환합니다.
    
    Args:
        text (str): 임베딩으로 변환할 문장.
        model_name (str): 사용할 임베딩 모델 이름 (기본값: gemini-embedding-2-preview).
    
    Returns:
        list[float]: 변환된 임베딩 벡터.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    vector = embeddings.embed_query(text)
    return vector

def split_pdf_document(file_path: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    PDF 문서를 로드하고 지정된 크기 단위로 분할합니다.
    
    Args:
        file_path (str): 로드할 PDF 파일의 경로.
        chunk_size (int): 분할할 청크의 크기 (기본값: 800).
        chunk_overlap (int): 청크 간 중복(Overlap) 크기 (기본값: 100).
        
    Returns:
        List[Document]: 분할된 Document 객체 리스트.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"해당 경로에서 파일을 찾을 수 없습니다: {file_path}")

    # 1. PDF 문서 로드
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Text Splitter 초기화
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 3. 문서 분할 및 반환
    split_docs = text_splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    # 이 파일이 직접 실행될 경우의 테스트 코드 (필요 시 수정하여 사용하세요)
    print("utils 모듈 테스트입니다.")
    # result = split_pdf_document("CS매뉴얼_25년_개정판.pdf")
    # vector = generate_sentence_embedding("hello, world!")
    # print(vector[:5])
