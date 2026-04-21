from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv(".env")  # .env 파일에서 환경 변수 로드
api_key = os.getenv("GOOGLE_API_KEY")  # Google API 키를 안전하게 가져오기

# 모델 객체 생성 (API 키는 환경변수 GOOGLE_API_KEY에 있어야 함)
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in seoul with korean"}]}
)
# Extract final text
final_message = result['messages'][-1]
content = final_message.content
if isinstance(content, list):
    text = content[0].get('text', '')
else:
    text = content

print(text)