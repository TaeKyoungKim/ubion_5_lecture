import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_pdf_document(file_path: str):
    """
    PDF 문서를 로드하고 지정된 크기 단위로 분할합니다.
    """
    # 1. PDF 문서 로드
    # pip install pypdf 를 먼저 실행해야 할 수 있습니다.
    print(f"[{file_path}] 문서를 로드하는 중...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Text Splitter 초기화
    # tiktoken 기반으로 텍스트를 분할합니다. 
    # pip install tiktoken 설치가 필요할 수 있습니다.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100
    )
    
    # 3. 문서 분할 (Document 객체 리스트를 분할할 때는 split_documents 사용)
    split_docs = text_splitter.split_documents(documents)
    
    print(f"총 {len(split_docs)}개의 청크(chunk)로 분할되었습니다.\n")
    
    # 4. 결과 출력 (처음 3개만 예시로 출력)
    for i, doc in enumerate(split_docs[:3]):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}\n")
        
    return split_docs

if __name__ == "__main__":
    # 테스트용 PDF 파일 경로를 지정하세요. (예: "sample.pdf")
    result = split_pdf_document(f"utils/CS매뉴얼_25년_개정판.pdf")
    print(result)
    # pass
