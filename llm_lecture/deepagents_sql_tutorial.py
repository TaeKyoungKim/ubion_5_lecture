import os
import requests
from dotenv import load_dotenv

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

# 환경 변수 로드 (명시적으로 키 설정)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9XE9vYyYjaNOHIux0q_RVyed_mlVhtq4" 
# 모델 설정
MODEL_NAME = "gemini-3.1-pro-preview"

def setup_database():
    """Chinook.db가 없으면 자동으로 다운로드합니다."""
    db_path = "chinook.db"
    if not os.path.exists(db_path):
        print("Chinook 데이터베이스 다운로드 중...")
        url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
        response = requests.get(url)
        with open(db_path, 'wb') as f:
            f.write(response.content)
        print("다운로드 완료!")
    return db_path

def main():
    # 1. DB 준비
    db_path = setup_database()
    
    # 2. SQLDatabase 연결 설정
    # 사용할 샘플 로우 갯수를 3으로 한정하여 프롬프트 토큰 절약
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
    
    # 3. 툴킷을 위한 Langchain LLM 인스턴스 초기화 (쿼리 오류 수정 등의 툴 내부 로직에서 사용됨)
    # ChatGoogleGenAI를 호환을 위해 사용합니다. 
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    
    # SQLDatabaseToolkit을 이용해 유용한 SQL 도구(테이블 조회, 쿼리 수행 등)들을 자동으로 추출
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()
    
    # 4. 에이전트 지시 사항 (System Prompt) 지정 
    # **한국어 답변을 강제**
    sql_agent_instructions = """당신은 데이터베이스 분석 전문가입니다.
보유한 SQL 도구들을 활용하여 데이터베이스의 테이블 구조를 파악하고, 사용자 질문에 알맞은 올바른 SQL 쿼리를 작성하여 실행해야 합니다.

## 검색 및 실행 가이드
1. 맨 처음 `list_tables` 나 `get_schema` 도구를 사용해 질문과 관련된 테이블을 찾으세요.
2. 데이터를 검색할 `execute_query` 도구를 써서 결과를 가져오세요.

## 출력 지시사항
- **모든 설명과 최종 결과 도출은 반드시 한국어(Korean)로 작성하세요.**
- 최종 답변에는 숫자가 의미하는 바에 대한 분석이나 통찰력을 포함하세요.
- 데이터베이스 쿼리를 짜는 플랜 과정이나 사고 과정을 상세히 보여주면 더욱 좋습니다.
"""

    # 5. Deep Agent 생성
    agent = create_deep_agent(
        model=f"google_genai:{MODEL_NAME}", # deepagents 용 litellm 포맷
        tools=sql_tools,
        system_prompt=sql_agent_instructions,
    )
    
    print("\n[에이전트 준비 완료. 질문을 던집니다...]\n")
    
    # 한글로 묻는 예시 질문
    question = "가장 많은 매출을 발생시킨 최고 실적 직원(Employee)은 누구이며, 주 1위 매출 국가(Country)는 어디인가요?"
    print(f"질문: {question}\n")
    
    # 에이전트 실행
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
    # 최종 답변 출력
    print("="*50)
    print("답변 내용:")
    print("="*50)
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
