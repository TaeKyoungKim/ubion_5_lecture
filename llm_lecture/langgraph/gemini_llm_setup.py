import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# .env 파일에서 환경변수(ex: GOOGLE_API_KEY)를 읽어옵니다.
load_dotenv()

# Google Gemini API Key가 정상적으로 로드되었는지 검증
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("❌ .env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")

# Gemini 모델 객체 생성 (기본적으로 빠르고 우수한 'gemini-2.5-flash'를 세팅함)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Ollama 모델 객체 생성 (기본적으로 빠르고 우수한 'llama3.2:1b'를 세팅함)
ollama_llm = ChatOllama(model="llama3.2:1b")

if __name__ == "__main__":
    query = input("어떤모델을 사용하시겠습니까? (1: Gemini, 2: Ollama): ")
    if query == "1":
        llm = gemini_llm
        print(f"✅ Gemini 모델 객체가 성공적으로 생성되었습니다: {llm.model}")
    else:
        llm = ollama_llm
        print(f"✅ Ollama 모델 객체가 성공적으로 생성되었습니다: {llm.model}")
    query = input("질문을 입력하세요: ")
    response = llm.invoke(query)
    print(response.content)
