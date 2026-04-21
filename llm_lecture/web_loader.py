from langchain_community.document_loaders import WebBaseLoader

# 1. 로드하고자 하는 LLM 관련 설명 사이트 URL 리스트
urls = [
    "https://python.langchain.com/docs/introduction/",             # LangChain 소개
    "https://www.anthropic.com/claude",                            # Anthropic Claude 설명
    "https://openai.com/blog/chatgpt"                              # OpenAI ChatGPT 블로그
]

# 2. WebBaseLoader 초기화 (여러 URL을 한 번에 전달 가능)
loader = WebBaseLoader(urls)

# 3. 데이터 로드
# load() 메서드는 각 URL의 내용을 Document 객체 리스트로 반환합니다.
docs = loader.load()

# 4. 결과 출력
print(f"로드된 문서 개수: {len(docs)}\n")

for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Title: {doc.metadata.get('title', 'No Title')}")
    # 내용이 너무 길 수 있으므로 앞의 200자만 출력
    print(f"Content Snippet: {doc.page_content[:200].strip()}...")
    print("\n")