import os
import requests
from datetime import datetime
import pytz

# 최신 LangChain v1.0+ 핵심 모듈
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# .env 파일에서 환경변수(ex: GOOGLE_API_KEY)를 읽어옵니다.
load_dotenv()

# Google Gemini API Key가 정상적으로 로드되었는지 검증
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("❌ .env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")


# 1. 기초 설정 (시간 및 색상)
KST = pytz.timezone('Asia/Seoul')
C = {
    "bold": "\033[1m", "reset": "\033[0m", 
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m"
}

# 2. 실시간 데이터 수집 도구
@tool
def fetch_market_data(query: str = "") -> dict:
    """Binance Futures + Fear&Greed + CoinGecko 데이터를 수집하여 딕셔너리로 반환합니다."""
    b, r, cy = C["bold"], C["reset"], C["cyan"]

    print(f"\n{cy}{b}{'─'*60}")
    print(f"  Gemini가 시장 데이터를 수집 중입니다... (기준: KST)")
    print(f"{'─'*60}{r}")

    data = {
        "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        "price": 0.0, "price_1h_ago": 0.0, "change_24h": 0.0,
        "funding_rate": 0.0, "open_interest": 0.0,
        "fear_greed": 50, "fear_greed_label": "Neutral", "dominance": 50.0,
    }

    try:
        # Binance Futures 24H & 1H Kline
        d = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT", timeout=8).json()
        data.update({"price": float(d["lastPrice"]), "change_24h": float(d["priceChangePercent"])})
        
        k = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=2", timeout=8).json()
        if k and len(k) >= 2: data["price_1h_ago"] = float(k[-2][4])

        # 펀딩비 & OI
        fd = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=8).json()
        data["funding_rate"] = float(fd["lastFundingRate"]) * 100
        oi = requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT", timeout=8).json()
        data["open_interest"] = float(oi["openInterest"])

        # Fear & Greed & Dominance
        fg = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        data["fear_greed"], data["fear_greed_label"] = int(fg["value"]), fg["value_classification"]
        dom = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()["data"]["market_cap_percentage"]["btc"]
        data["dominance"] = round(dom, 2)

        print(f"  {C['green']}✓{r} 모든 지표 수집 완료")
    except Exception as e:
        print(f"  {C['yellow']}⚠ 오류 발생: {e}{r}")

    return data

# 3. 모델 및 도구 설정 (API 키는 OS 환경변수 GOOGLE_API_KEY 에 등록되어 있어야 함)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
tools = [fetch_market_data]

# 4. 에이전트 생성 (AgentExecutor 완전 제거, create_agent만 사용)
# 내부적으로 LangGraph 상태 관리(State Graph)를 통해 자동으로 도구를 호출합니다.
agent = create_agent(model=model, tools=tools)

# 5. 실행부
if __name__ == "__main__":
    SYSTEM_PROMPT = """너는 30년이상의 경력을 가진 전설적인 암호화폐 트레이더이다.
주어진 데이터를 기반으로 smc(ict)기법을 활용하여 분석하고 리포트를 작성해줘.
롱관점 숏관점을 모두 제시하고, 진입가 , 손절가, 익절가도 제시해
"""

    query = "지금 현재 비트코인선물 가격을 기준으로 주요 지표들 분석해서 대응시나리오"
    
    # 최신 규격은 'messages' 배열로 시스템 프롬프트와 사용자 입력을 한 번에 넘깁니다.
    response = agent.invoke({
        "messages": [
            ("system", SYSTEM_PROMPT),
            ("user", query)
        ]
    })
    
    # LangGraph 기반 에이전트의 결과물은 대화 기록 전체(messages 배열)로 반환됩니다.
    # 배열의 가장 마지막 메시지가 최종 AI의 답변입니다.
    final_answer = response["messages"][-1].content
    
    print(f"\n{C['cyan']}{C['bold']}[Gemini AI 시장 리포트]{C['reset']}\n{final_answer}")
