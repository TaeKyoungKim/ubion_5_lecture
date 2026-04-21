import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 환경 변수 로드 (.env 파일에서 GOOGLE_API_KEY 가져오기)
load_dotenv(".env")
api_key = os.getenv("GOOGLE_API_KEY")  # Google API 키를 안전하게 가져오기
def generate_sentence_embedding(text: str):
    """
    임의의 문장을 받아서 벡터 임베딩을 생성하고 결과를 출력합니다.
    """
    print(f"[입력된 문장]: \"{text}\"\n")
    
    # 1. 임베딩 모델 초기화
    # 질문에 명시해주신 모델을 사용합니다.
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # 2. 문장을 벡터로 변환 (단일 문장/쿼리에 대해 변환 시 embed_query를 사용)
    print("벡터 임베딩을 생성하는 중...")
    vector = embeddings.embed_query(text)
    
    # 3. 변환된 벡터 정보 출력
    # 임베딩 벡터는 매우 긴 실수 배열이므로 차원수와 앞부분 일부만 출력합니다.
    print(f"✅ 변환 완료!")
    print(f"👉 벡터의 차원 수 (Dimension): {len(vector)}")
    print(f"👉 벡터 값 일부 (처음 5개 요소): {vector[:5]}")
    
    return vector

if __name__ == "__main__":
    print("==================================================")
    print("Google Gemini Embedding Test")
    print("==================================================")
    
    # 사용자로부터 터미널 창에서 직접 입력을 받습니다.
    user_input = input("벡터로 변환할 문장을 입력해보세요 (빈 칸으로 엔터 시 기본값 사용): ")
    
    if not user_input.strip():
        # 빈 값 입력 시 기본 테스트 문장 사용
        user_input = "hello, world!"
        
    generate_sentence_embedding(user_input)

