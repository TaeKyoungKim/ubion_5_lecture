from langchain_community.document_loaders import PyPDFLoader
from typing import List, Union
from langchain_core.documents import Document

class PdfDocumentLoader:
    """
    여러 PDF 파일의 경로를 받아 내용을 로드하고 관리하는 유틸리티 클래스입니다.
    """
    def __init__(self, file_paths: Union[str, List[str]] = None):
        """
        초기화 시 로드할 PDF 경로를 받을 수 있습니다 (선택사항).
        비워두었다가 load() 메서드를 호출할 때 직접 경로를 넘겨줄 수도 있습니다.
        
        Args:
            file_paths (Union[str, List[str]], optional): 기본으로 설정할 PDF 파일 경로 혹은 경로 목록
        """
        if file_paths is None:
            self.file_paths = []
        elif isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths

    def load(self, file_paths: Union[str, List[str]] = None) -> List[Document]:
        """
        설정된 파일 경로, 혹은 직접 전달받은 파일 경로로부터 PDF 데이터를 로드합니다.
        여러 개의 PDF가 주어진 경우, 각각 로드하여 하나의 리스트로 합쳐 반환합니다.
        
        Args:
            file_paths (Union[str, List[str]], optional): 로드할 PDF 파일 경로 혹은 목록. 
                                        값을 넘기면 그 경로를 사용하고, 안 넘기면 인스턴스 생성 시점의 값을 씁니다.
        Returns:
            List[Document]: 로딩 완료된 페이지별 Document 객체의 리스트
        """
        target_paths = file_paths if file_paths is not None else self.file_paths
        
        if not target_paths:
            print("경고: 로드할 PDF 파일 경로가 없습니다. 경로를 전달해주세요.")
            return []
            
        # 만약 단일 문자열로 들어오면 리스트로 변환
        if isinstance(target_paths, str):
            target_paths = [target_paths]
            
        all_docs = []
        for path in target_paths:
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"[{path}] 로드 중 오류가 발생했습니다: {e}")
                
        return all_docs

    def print_docs_info(self, docs: List[Document], snippet_length: int = 200) -> None:
        """
        로드된 문서들의 메타데이터와 내용 일부(스니펫)를 터미널에 깔끔하게 출력합니다.
        
        Args:
            docs (List[Document]): 출력할 문서 객체 리스트
            snippet_length (int): 출력할 페이지 내용의 길이 제한 (기본값: 200자)
        """
        print(f"로드된 PDF 페이지(문서) 총 개수: {len(docs)}\n")

        for i, doc in enumerate(docs):
            print(f"--- Page {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page Number: {doc.metadata.get('page', 'Unknown')}")
            # 불필요한 공백과 줄바꿈을 지우고 깔끔하게 출력
            content = " ".join(doc.page_content.split())
            print(f"Content Snippet: {content[:snippet_length].strip()}...\n")


if __name__ == "__main__":
    # 유틸리티 클래스 테스트 및 사용 예시
    
    # 1. 인스턴스 생성 (초기 경로 없이)
    pdf_loader_util = PdfDocumentLoader()
    
    # 예시용 PDF 파일 경로 (실제로 존재하는 파일로 변경해야 동작합니다)
    sample_pdf_path = "./lorem-ipsum-10pages.pdf"
    
    # 2. 문서 로드 (경로 전달)
    print(f"[{sample_pdf_path}] PDF 로드 시도 중...")
    
    # 파일이 존재하지 않으면 예외 처리가 발생하므로 안내문 출력
    print("※ (주의) 해당 경로에 실제 PDF 파일이 있어야 페이지가 정상 반환됩니다.\n")
    
    loaded_pages = pdf_loader_util.load(file_paths=sample_pdf_path)
    
    # 3. 로드된 문서 정보 출력
    if loaded_pages:
        pdf_loader_util.print_docs_info(loaded_pages)
