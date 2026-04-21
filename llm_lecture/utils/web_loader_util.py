from langchain_community.document_loaders import WebBaseLoader
from typing import List
from langchain_core.documents import Document

class WebDocumentLoader:
    """
    여러 웹사이트의 URL을 받아 내용을 로드하고 관리하는 유틸리티 클래스입니다.
    """
    def __init__(self, urls: List[str] = None):
        """
        초기화 시 로드할 URL 리스트를 받을 수 있습니다 (선택사항).
        비워두었다가 load() 메서드를 호출할 때 직접 URL을 넘겨줄 수도 있습니다.
        
        Args:
            urls (List[str], optional): 기본으로 설정할 웹사이트 URL 목록
        """
        self.urls = urls if urls is not None else []

    def load(self, urls: List[str] = None) -> List[Document]:
        """
        설정된 URL, 혹은 직접 전달받은 URL 리스트로부터 웹 페이지의 모든 데이터를 로드합니다.
        
        Args:
            urls (List[str], optional): 로드할 웹사이트 URL 목록. 
                                        값을 넘기면 그 URL을 사용하고, 안 넘기면 인스턴스 생성 시점의 값을 씁니다.
        Returns:
            List[Document]: 로딩 완료된 Document 객체의 리스트
        """
        target_urls = urls if urls is not None else self.urls
        
        if not target_urls:
            print("경고: 로드할 URL이 없습니다. URL 리스트를 전달해주세요.")
            return []
            
        loader = WebBaseLoader(target_urls)
        return loader.load()

    def print_docs_info(self, docs: List[Document], snippet_length: int = 200) -> None:
        """
        로드된 문서들의 메타데이터와 내용 일부(스니펫)를 터미널에 깔끔하게 출력합니다.
        
        Args:
            docs (List[Document]): 출력할 문서 객체 리스트
            snippet_length (int): 출력할 첫 페이지 내용의 길이 제한 (기본값: 200자)
        """
        print(f"로드된 문서 개수: {len(docs)}\n")

        for i, doc in enumerate(docs):
            print(f"--- Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Title: {doc.metadata.get('title', 'No Title')}")
            # 내용이 너무 길 수 있으므로 snippet_length 이하로만 출력
            print(f"Content Snippet: {doc.page_content[:snippet_length].strip()}...\n")


if __name__ == "__main__":
    # 유틸리티 클래스 테스트 및 사용 예시
    sample_urls = [
        "https://python.langchain.com/docs/introduction/",             
        "https://www.anthropic.com/claude",                            
        "https://openai.com/blog/chatgpt"                              
    ]
    
    # 1. 인스턴스 생성 (이때 URL을 세팅하지 않아도 됩니다.)
    web_loader_util = WebDocumentLoader()
    
    # 2. 문서 로드 시점에 URL을 원하는 대로 전달합니다.
    print("웹사이트 데이터 로드 시작...")
    loaded_docs = web_loader_util.load(urls=sample_urls)
    
    # 3. 로드된 문서 정보 출력
    if loaded_docs:
        web_loader_util.print_docs_info(loaded_docs)
