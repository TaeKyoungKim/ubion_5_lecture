from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from tavily import TavilyClient

import os
from dotenv import load_dotenv
import datetime

# .env 파일을 읽어옴
load_dotenv()

# os.environ을 통해 값에 접근
google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_api_key
print(f"API Key: {google_api_key}")
tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

# response = tavily_client.search("Who is Leo Messi?")

# print(response)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Gemini defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

@tool
def search(query: str) -> str:
    """Search for information."""
    response = tavily_client.search(query)
    return f"Results for: {response}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: raining, 100°F"

@tool
def indicator(method: str) -> str:
    """Analyze the stock market."""
    return f"Analysis result: 매도"

@tool
def get_today_date() -> str:
    """Get the current today's date."""
    now = datetime.datetime.now()
    return now.strftime("%Y년 %m월 %d일")

agent = create_agent(
    model, 
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "어린아이를 대상으로 답변해줘",
            }
        ]
    ),
    tools=[search, get_weather, indicator, get_today_date])

query = input("질문을 입력하세요: ")

result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]}
)
# 최종 답변만 깔끔하게 출력하기
final_message = result['messages'][-1]
if isinstance(final_message.content, list):
    print("답변:", final_message.content[0].get('text', ''))
else:
    print("답변:", final_message.content)