"""
Bitcoin 선물 병렬 분석기 v2.1
LangGraph 병렬 패턴 + 실시간 데이터 자동 수집 + 다각도 트레이딩 플랜

개선사항 (v2 → v2.2):
  1. response_mime_type="application/json" → Gemini JSON 출력 강제
  2. safe_get_text() 추가 → response.text가 None/빈문자열일 때 candidates 직접 순회
  3. TRADE_JSON_SCHEMA의 '숫자' 플레이스홀더 → 실제 예시 숫자로 교체
  4. parse_json() brace-counting 알고리즘으로 교체 (greedy regex 제거)
  5. call_llm() try-except + 자동 재시도 (최대 3회)
  6. 각 LLM 노드 try-except 추가
  7. DEBUG_LLM 옵션 추가 (원본 응답 확인용)

실행:
  pip install google-genai langgraph requests
  Windows : set GOOGLE_API_KEY=AIza...
  Mac/Linux: export GOOGLE_API_KEY=AIza...
  python btc_smc_parallel_v2_fixed.py
"""

import json
import os
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# .env 파일에서 환경변수 로드
load_dotenv()

# ─────────────────────────────────────────────
# API 및 모델 설정
# ─────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "\n[오류] GOOGLE_API_KEY 환경변수 미설정\n"
        "  Windows : set GOOGLE_API_KEY=AIza...\n"
        "  Mac/Linux: export GOOGLE_API_KEY=AIza...\n"
    )

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.5-flash"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_llm = None
if GROQ_API_KEY:
    try:
        groq_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="qwen/qwen3-32b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
    except Exception as e:
        print(f"[경고] Groq 초기화 실패: {e}")

SELECTED_MODEL = "gemini"

# True 로 변경하면 LLM 원본 응답을 출력 (파싱 실패 원인 디버깅용)
DEBUG_LLM = False

KST = timezone(timedelta(hours=9))

C = {
    "reset":  "\033[0m",  "bold":   "\033[1m",
    "green":  "\033[92m", "red":    "\033[91m",
    "yellow": "\033[93m", "blue":   "\033[94m",
    "cyan":   "\033[96m", "purple": "\033[95m",
    "gray":   "\033[90m", "white":  "\033[97m",
}


# ─────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────
class State(TypedDict):
    # ── 자동 수집 데이터 ──────────────────────
    timestamp:        str
    price:            float
    price_1h_ago:     float
    change_24h:       float
    high_24h:         float
    low_24h:          float
    volume_24h:       float
    funding_rate:     float
    open_interest:    float
    fear_greed:       int
    fear_greed_label: str
    dominance:        float

    # ── 사용자 추가 입력 ──────────────────────
    capital:          float
    leverage:         int
    timeframe:        str
    extra_info:       str

    # ── LLM 노드 출력 ─────────────────────────
    smc_plan:     dict
    ta_plan:      dict
    risk_plan:    dict

    # ── Aggregator 출력 ───────────────────────
    final_plan:   dict


# ─────────────────────────────────────────────
# 실시간 데이터 수집
# ─────────────────────────────────────────────
def fetch_market_data() -> dict:
    b, r = C["bold"], C["reset"]
    cy = C["cyan"]

    print(f"\n{cy}{b}{'─'*60}")
    print(f"  실시간 시장 데이터 수집 중...")
    print(f"{'─'*60}{r}")

    data = {
        "timestamp":        datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        "price":            0.0,
        "price_1h_ago":     0.0,
        "change_24h":       0.0,
        "high_24h":         0.0,
        "low_24h":          0.0,
        "volume_24h":       0.0,
        "funding_rate":     0.0,
        "open_interest":    0.0,
        "fear_greed":       50,
        "fear_greed_label": "Neutral",
        "dominance":        50.0,
    }

    # ── Binance Futures 24H 티커 ──
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT"
        resp = requests.get(url, timeout=8)
        d = resp.json()
        data["price"]       = float(d["lastPrice"])
        data["change_24h"]  = float(d["priceChangePercent"])
        data["high_24h"]    = float(d["highPrice"])
        data["low_24h"]     = float(d["lowPrice"])
        data["volume_24h"]  = float(d["volume"])
        print(f"  {C['green']}✓{r} Binance Futures 가격: ${data['price']:,.2f}")
    except Exception as e:
        print(f"  {C['yellow']}⚠ Binance Futures 오류 ({e}) → 수동 입력 필요{r}")

    # ── Binance 1H 캔들 ──
    try:
        url2 = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=2"
        resp2 = requests.get(url2, timeout=8)
        klines = resp2.json()
        if klines and len(klines) >= 2:
            data["price_1h_ago"] = float(klines[-2][4])
        print(f"  {C['green']}✓{r} 1시간 전 가격: ${data['price_1h_ago']:,.2f}")
    except Exception as e:
        print(f"  {C['yellow']}⚠ 1H 캔들 오류 ({e}){r}")

    # ── 펀딩비 ──
    try:
        url3 = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
        resp3 = requests.get(url3, timeout=8)
        fd = resp3.json()
        data["funding_rate"] = float(fd["lastFundingRate"]) * 100
        print(f"  {C['green']}✓{r} 펀딩비: {data['funding_rate']:+.4f}%")
    except Exception as e:
        print(f"  {C['yellow']}⚠ 펀딩비 오류 ({e}){r}")

    # ── 미결제약정 (OI) ──
    try:
        url4 = "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT"
        resp4 = requests.get(url4, timeout=8)
        oi = resp4.json()
        data["open_interest"] = float(oi["openInterest"])
        print(f"  {C['green']}✓{r} 미결제약정: {data['open_interest']:,.0f} BTC")
    except Exception as e:
        print(f"  {C['yellow']}⚠ OI 오류 ({e}){r}")

    # ── Fear & Greed Index ──
    try:
        url5 = "https://api.alternative.me/fng/?limit=1"
        resp5 = requests.get(url5, timeout=8)
        fg = resp5.json()["data"][0]
        data["fear_greed"]       = int(fg["value"])
        data["fear_greed_label"] = fg["value_classification"]
        print(f"  {C['green']}✓{r} 공포탐욕지수: {data['fear_greed']} ({data['fear_greed_label']})")
    except Exception as e:
        print(f"  {C['yellow']}⚠ Fear&Greed 오류 ({e}){r}")

    # ── BTC 도미넌스 ──
    try:
        url6 = "https://api.coingecko.com/api/v3/global"
        resp6 = requests.get(url6, timeout=10)
        dom = resp6.json()["data"]["market_cap_percentage"]["btc"]
        data["dominance"] = round(dom, 2)
        print(f"  {C['green']}✓{r} BTC 도미넌스: {data['dominance']}%")
    except Exception as e:
        print(f"  {C['yellow']}⚠ 도미넌스 오류 ({e}){r}")

    return data


def manual_input_price(auto_data: dict) -> dict:
    b, r = C["bold"], C["reset"]
    if auto_data["price"] == 0.0:
        print(f"\n  {b}BTC 선물 현재가를 직접 입력하세요{r}")
        try:
            auto_data["price"] = float(input("  BTC 선물가 (USD): ").strip())
        except ValueError:
            auto_data["price"] = 84500.0
    return auto_data


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

# ★ FIX A: response.text가 None/빈문자열일 때 candidates 직접 순회
def safe_get_text(response) -> str:
    """
    google-genai 1.x에서 response.text가 None 또는 ''을 반환하는 경우 대비.
    thinking 모델(gemini-2.5-*) 에서 자주 발생.
    1) response.text 우선 시도
    2) 실패 시 candidates[0].content.parts를 직접 순회해 텍스트 수집
       (thought=True 파트는 제외)
    """
    # 1단계: 표준 접근
    try:
        t = response.text
        if t:           # None, '' 모두 걸러냄
            return t
    except Exception:
        pass

    # 2단계: candidates 직접 순회 (thought 파트 제외)
    try:
        for candidate in (response.candidates or []):
            parts = getattr(candidate.content, "parts", None) or []
            texts = [
                p.text for p in parts
                if getattr(p, "text", None)
                and not getattr(p, "thought", False)   # thinking 파트 제외
            ]
            if texts:
                return " ".join(texts)
    except Exception:
        pass

    return ""   # 완전 실패


# ★ FIX B: response_mime_type="application/json" + 재시도 + safe_get_text
def call_llm(system: str, user: str, max_tokens: int = 1000,
             retries: int = 3, delay: float = 2.0) -> str:
    """
    Gemini / Groq 선택 호출.
    - safe_get_text()로 None/빈 응답 안전 처리
    - API 오류 시 최대 retries 회 자동 재시도
    """
    global SELECTED_MODEL

    if SELECTED_MODEL == "groq" and groq_llm:
        for attempt in range(1, retries + 1):
            try:
                messages = [
                    SystemMessage(content=system),
                    HumanMessage(content=user)
                ]
                response = groq_llm.invoke(messages)
                text = response.content
                if text:
                    return text
                print(f"  {C['yellow']}⚠ 빈 응답 수신 (시도 {attempt}/{retries}) → 재시도{C['reset']}")
            except Exception as e:
                if attempt < retries:
                    print(f"  {C['yellow']}⚠ Groq API 오류 (시도 {attempt}/{retries}): {e}"
                          f" → {delay:.0f}초 후 재시도{C['reset']}")
                    time.sleep(delay)
                else:
                    print(f"  {C['red']}✗ Groq API 오류 (재시도 초과): {e}{C['reset']}")
        return ""
        
    else:
        for attempt in range(1, retries + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=MODEL,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=0.2,
                        response_mime_type="application/json",  # ★ JSON 강제
                    ),
                )
                text = safe_get_text(response)
                if text:
                    return text
                # 빈 응답 → 재시도
                print(f"  {C['yellow']}⚠ 빈 응답 수신 (시도 {attempt}/{retries}) → 재시도{C['reset']}")
    
            except Exception as e:
                if attempt < retries:
                    print(f"  {C['yellow']}⚠ Gemini API 오류 (시도 {attempt}/{retries}): {e}"
                          f" → {delay:.0f}초 후 재시도{C['reset']}")
                    time.sleep(delay)
                else:
                    print(f"  {C['red']}✗ Gemini API 오류 (재시도 초과): {e}{C['reset']}")
    
        return ""   # 모든 재시도 실패 → 빈 문자열 (parse_json이 기본값 처리)


def build_context(state: State) -> str:
    price_change_1h = ""
    if state.get("price_1h_ago", 0) > 0:
        chg = (state["price"] - state["price_1h_ago"]) / state["price_1h_ago"] * 100
        price_change_1h = f" | 1H 변동: {chg:+.2f}%"

    return f"""
[시장 데이터] {state['timestamp']}
BTC 선물가: ${state['price']:,.2f}
24H 변동: {state['change_24h']:+.2f}%{price_change_1h}
24H 고가: ${state['high_24h']:,.2f} | 24H 저가: ${state['low_24h']:,.2f}
24H 거래량: {state['volume_24h']:,.0f} BTC
펀딩비: {state['funding_rate']:+.4f}%
미결제약정: {state['open_interest']:,.0f} BTC
공포탐욕지수: {state['fear_greed']}/100 ({state['fear_greed_label']})
BTC 도미넌스: {state['dominance']:.2f}%
분석 타임프레임: {state['timeframe']}
분석 자본금: ${state['capital']:,.0f} USDT | 레버리지: {state['leverage']}x
추가정보: {state.get('extra_info', '없음')}""".strip()


TRADE_JSON_SCHEMA = """{
  "direction": "LONG 또는 SHORT 또는 WAIT",
  "confidence": "상 또는 중 또는 하",
  "entry":        84000,
  "sl":           83000,
  "tp1":          85000,
  "tp2":          86000,
  "tp3":          88000,
  "entry_reason": "진입 근거 (80자)",
  "sl_reason":    "손절 근거 (50자)",
  "tp1_reason":   "1차 익절 근거 (50자)",
  "tp2_reason":   "2차 익절 근거 (50자)",
  "tp3_reason":   "3차 익절 근거 (50자)",
  "invalidation": "무효화 조건 (50자)",
  "key_levels":   ["85000", "83000", "82000"],
  "summary":      "핵심 요약 (100자)"
}"""


# ★ FIX 2: brace-counting 기반 parse_json (greedy regex 대체)
def _extract_json_by_braces(text: str) -> str | None:
    """
    중괄호 균형을 직접 세어 첫 번째 완전한 JSON 객체를 추출.
    greedy regex({.*})가 trailing text를 포함하는 문제를 해결.
    후행 콤마({...,}) 자동 수정 포함.
    """
    brace_count = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start != -1:
                candidate = text[start:i + 1]
                # 직접 파싱 시도
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    # 후행 콤마 제거 후 재시도
                    fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
                    try:
                        json.loads(fixed)
                        return fixed
                    except Exception:
                        # 이 블록 포기, 다음 { 탐색
                        start = -1
    return None


def parse_json(raw: str, label: str) -> dict:
    """
    3단계 파싱 전략:
      1) 마크다운 제거 후 직접 파싱
      2) brace-counting으로 JSON 블록 추출
      3) 기본값(WAIT) 반환
    """
    if DEBUG_LLM:
        print(f"\n  {C['gray']}[DEBUG {label}] 원본 응답 ({len(raw)}자):\n"
              f"{raw[:600]}{C['reset']}")

    # 마크다운 코드 블록 제거 (```json ... ``` / ``` ... ```)
    clean = re.sub(r'```(?:json)?\s*', '', raw)
    clean = clean.replace('```', '').strip()

    # 1단계: 직접 파싱
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 2단계: brace-counting 추출
    candidate = _extract_json_by_braces(clean)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3단계: 실패 → 기본값
    print(f"  {C['yellow']}⚠ [{label}] JSON 파싱 실패 → 기본값 사용"
          f"  (DEBUG_LLM=True 로 원본 확인 가능){C['reset']}")
    return {
        "direction": "WAIT", "confidence": "하",
        "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0,
        "entry_reason": f"[{label}] 파싱 실패",
        "sl_reason": "", "tp1_reason": "", "tp2_reason": "", "tp3_reason": "",
        "invalidation": "", "key_levels": [], "summary": f"[{label}] 파싱 실패"
    }


# ─────────────────────────────────────────────
# LLM 노드 1: SMC 관점
# ─────────────────────────────────────────────
def call_llm_1(state: State) -> dict:
    print(f"\n  {C['blue']}[LLM-1] SMC 구조 분석 중...{C['reset']}")

    system = f"""당신은 ICT/SMC(Smart Money Concepts) 전문 비트코인 선물 트레이더입니다.
실시간 시장 데이터를 기반으로 SMC 관점에서 트레이딩 플랜을 수립하세요.

분석 기준:
- Order Block (OB): 가격 근처 기관 매수/매도 블록
- Fair Value Gap (FVG): 24H 고/저가 기준 미충족 구간
- Liquidity: 24H 고가 위 / 저가 아래 유동성 풀
- Market Structure: 추세 방향 (BOS / CHoCH)
- Premium(61.8% 이상) / Discount(38.2% 이하) Zone
- 펀딩비 방향성: 포지션 편중 확인

반드시 순수 JSON만 출력하세요:
{TRADE_JSON_SCHEMA}

숫자 규칙: 정수(USD), LONG이면 sl<entry<tp1<tp2<tp3, SHORT이면 sl>entry>tp1>tp2>tp3"""

    # ★ FIX 3: 노드별 try-except
    try:
        raw = call_llm(system, build_context(state))
        result = parse_json(raw, "SMC")
    except Exception as e:
        print(f"  {C['red']}✗ [LLM-1] 예외: {e}{C['reset']}")
        result = parse_json("{}", "SMC")

    print(f"  {C['blue']}[LLM-1] 완료 → {result.get('direction')} "
          f"(신뢰도: {result.get('confidence')}){C['reset']}")
    return {"smc_plan": result}


# ─────────────────────────────────────────────
# LLM 노드 2: 기술적 분석 관점
# ─────────────────────────────────────────────
def call_llm_2(state: State) -> dict:
    print(f"\n  {C['green']}[LLM-2] 기술적 분석 중...{C['reset']}")

    system = f"""당신은 비트코인 선물 기술적 분석(TA) 전문가입니다.
실시간 시장 데이터를 기반으로 기술적 분석 관점의 트레이딩 플랜을 수립하세요.

분석 기준:
- 가격 구조: 24H 고/저가, 지지/저항 레벨
- 추세 강도: 24H 변동률, 거래량 분석
- 모멘텀: 1H 가격 변화율
- 펀딩비: 롱/숏 포지션 편중 (양수=롱 과열, 음수=숏 과열)
- OI(미결제약정) 변화: 시장 참여도
- 공포탐욕지수: 극단값 역추세 신호
- 이동평균 추정: 24H 고저 기준 중간값

반드시 순수 JSON만 출력하세요:
{TRADE_JSON_SCHEMA}

숫자 규칙: 정수(USD), LONG이면 sl<entry<tp1<tp2<tp3, SHORT이면 sl>entry>tp1>tp2>tp3"""

    try:
        raw = call_llm(system, build_context(state))
        result = parse_json(raw, "TA")
    except Exception as e:
        print(f"  {C['red']}✗ [LLM-2] 예외: {e}{C['reset']}")
        result = parse_json("{}", "TA")

    print(f"  {C['green']}[LLM-2] 완료 → {result.get('direction')} "
          f"(신뢰도: {result.get('confidence')}){C['reset']}")
    return {"ta_plan": result}


# ─────────────────────────────────────────────
# LLM 노드 3: 리스크 관리 관점
# ─────────────────────────────────────────────
def call_llm_3(state: State) -> dict:
    print(f"\n  {C['yellow']}[LLM-3] 리스크 관리 분석 중...{C['reset']}")

    system = f"""당신은 비트코인 선물 리스크 관리 전문가입니다.
실시간 시장 데이터를 기반으로 리스크 관리 관점의 트레이딩 플랜을 수립하세요.

분석 기준:
- 변동성 위험: 24H 고저 범위 (ATR 추정)
- 레버리지 적정성: 현재 레버리지 vs 시장 변동성
- 펀딩비 리스크: 장기 보유 시 누적 비용
- 청산가 계산: entry × (1 ± 1/leverage) 기준
- 포지션 사이징: 자본금 × 리스크 % 기준
- 분할 진입/익절 전략
- 시나리오별 최대 손실 한도

반드시 순수 JSON만 출력하세요:
{TRADE_JSON_SCHEMA}

추가 필드를 JSON에 포함하세요:
  "liquidation_price": 82000,
  "max_loss_usdt": 500,
  "risk_reward_tp1": "1:2.0",
  "position_btc": 0.12,
  "split_entry": "분할 진입 전략 (60자)"

숫자 규칙: 정수(USD), LONG이면 sl<entry<tp1<tp2<tp3, SHORT이면 sl>entry>tp1>tp2>tp3"""

    try:
        raw = call_llm(system, build_context(state), max_tokens=1200)
        result = parse_json(raw, "Risk")
    except Exception as e:
        print(f"  {C['red']}✗ [LLM-3] 예외: {e}{C['reset']}")
        result = parse_json("{}", "Risk")

    print(f"  {C['yellow']}[LLM-3] 완료 → {result.get('direction')} "
          f"(신뢰도: {result.get('confidence')}){C['reset']}")
    return {"risk_plan": result}


# ─────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────
def aggregator(state: State) -> dict:
    print(f"\n  {C['purple']}[Aggregator] 3개 관점 통합 중...{C['reset']}")

    # 상위 노드 파싱 실패 여부 사전 경고
    failed = [
        label for key, label in [
            ("smc_plan", "SMC"), ("ta_plan", "TA"), ("risk_plan", "Risk")
        ]
        if "파싱 실패" in state.get(key, {}).get("entry_reason", "")
    ]
    if failed:
        print(f"  {C['yellow']}⚠ 파싱 실패 관점: {', '.join(failed)}"
              f" → aggregator 결과 신뢰도 낮음{C['reset']}")

    system = """당신은 비트코인 선물 수석 트레이딩 전략가입니다.
SMC / TA / 리스크관리 3가지 관점의 분석을 통합하여 최종 결론을 내리세요.

통합 규칙:
- 3개 중 2개 이상 같은 방향 → 해당 방향 채택
- 3개 모두 다른 방향 → WAIT
- 신뢰도: 3개 일치=상, 2개 일치=중, 불일치=하
- 진입가: 3개 entry 평균 (방향 일치 시)
- 손절가: 가장 보수적인 값 채택
- 익절가: 각 관점의 tp1/tp2/tp3 가중 평균

반드시 순수 JSON만 출력하세요:

{
  "direction":      "LONG 또는 SHORT 또는 WAIT",
  "confidence":     "상 또는 중 또는 하",
  "consensus":      "3개 관점 합의 요약 (80자)",
  "entry":          84000,
  "sl":             83000,
  "tp1":            85000,
  "tp2":            86000,
  "tp3":            88000,
  "liquidation":    82000,
  "position_size":  "권고 포지션 크기",
  "split_entry":    "분할 진입 전략 (60자)",
  "entry_reason":   "최종 진입 근거 (80자)",
  "sl_reason":      "최종 손절 근거 (50자)",
  "invalidation":   "무효화 조건 (50자)",
  "smc_verdict":    "SMC 관점 한줄 요약",
  "ta_verdict":     "TA 관점 한줄 요약",
  "risk_verdict":   "리스크 관점 한줄 요약",
  "action":         "즉시 실행 권고 1문장",
  "warnings":       ["주의사항1", "주의사항2"]
}"""

    user = (
        f"시장 데이터:\n{build_context(state)}\n\n"
        f"[SMC 관점]\n{json.dumps(state['smc_plan'],  ensure_ascii=False)}\n\n"
        f"[TA 관점]\n{json.dumps(state['ta_plan'],   ensure_ascii=False)}\n\n"
        f"[리스크 관점]\n{json.dumps(state['risk_plan'], ensure_ascii=False)}"
    )

    try:
        raw = call_llm(system, user, max_tokens=1500)
        result = parse_json(raw, "Aggregator")
    except Exception as e:
        print(f"  {C['red']}✗ [Aggregator] 예외: {e}{C['reset']}")
        result = parse_json("{}", "Aggregator")

    print(f"  {C['purple']}[Aggregator] 완료 → {result.get('direction')} "
          f"(신뢰도: {result.get('confidence')}){C['reset']}")
    return {"final_plan": result}


# ─────────────────────────────────────────────
# LangGraph 빌드 (이미지 패턴)
# ─────────────────────────────────────────────
def build_workflow() -> StateGraph:
    pb = StateGraph(State)
    pb.add_node("call_llm_1", call_llm_1)
    pb.add_node("call_llm_2", call_llm_2)
    pb.add_node("call_llm_3", call_llm_3)
    pb.add_node("aggregator", aggregator)

    pb.add_edge(START, "call_llm_1")
    pb.add_edge(START, "call_llm_2")
    pb.add_edge(START, "call_llm_3")
    pb.add_edge("call_llm_1", "aggregator")
    pb.add_edge("call_llm_2", "aggregator")
    pb.add_edge("call_llm_3", "aggregator")
    pb.add_edge("aggregator", END)

    return pb


# ─────────────────────────────────────────────
# 워크플로우 다이어그램
# ─────────────────────────────────────────────
def print_workflow_diagram():
    b, r = C["bold"], C["reset"]
    print(f"\n{b}{'─'*60}{r}")
    print(f"{b}  BTC 선물 병렬 분석기 v2.2  (LangGraph){r}")
    current_model_str = f"Groq (qwen/qwen3-32b)" if SELECTED_MODEL == "groq" else f"Gemini ({MODEL})  |  JSON 강제 모드: ON"
    print(f"  선택된 모델: {current_model_str}")
    print(f"{b}{'─'*60}{r}\n")
    print(f"   자동수집 → {b}[ START / In ]{r}")
    print(f"             ╱      │       ╲")
    print(f"            ▼       ▼        ▼")
    print(f"  {C['blue']}┌─────────┐{r}  {C['green']}┌─────────┐{r}  {C['yellow']}┌─────────┐{r}")
    print(f"  {C['blue']}│ LLM-1   │{r}  {C['green']}│ LLM-2   │{r}  {C['yellow']}│ LLM-3   │{r}")
    print(f"  {C['blue']}│ SMC     │{r}  {C['green']}│   TA    │{r}  {C['yellow']}│  Risk   │{r}")
    print(f"  {C['blue']}└─────────┘{r}  {C['green']}└─────────┘{r}  {C['yellow']}└─────────┘{r}")
    print(f"            ╲       │        ╱")
    print(f"             ▼      ▼       ▼")
    print(f"         {C['purple']}┌──────────────────┐{r}")
    print(f"         {C['purple']}│   Aggregator     │{r}")
    print(f"         {C['purple']}│  최종 종합 플랜  │{r}")
    print(f"         {C['purple']}└──────────────────┘{r}")
    print(f"                  │")
    print(f"                  ▼")
    print(f"           {b}[ END / Out ]{r}")
    print(f"{b}{'─'*60}{r}\n")


# ─────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────
def calc_rr(entry, sl, tp) -> str:
    try:
        risk = abs(entry - sl)
        return f"1:{abs(tp-entry)/risk:.1f}" if risk > 0 else "—"
    except Exception:
        return "—"


def print_plan_card(label: str, color: str, plan: dict, price: float):
    b, r = C["bold"], C["reset"]
    d = plan.get("direction", "WAIT")
    dc = C["green"] if d == "LONG" else C["red"] if d == "SHORT" else C["yellow"]
    entry = plan.get("entry", 0)
    sl    = plan.get("sl",    0)
    tp1   = plan.get("tp1",  0)
    tp2   = plan.get("tp2",  0)
    tp3   = plan.get("tp3",  0)

    print(f"\n  {color}{b}{'─'*54}{r}")
    print(f"  {color}{b}  {label}  →  {dc}{d}{color}  (신뢰도: {plan.get('confidence','—')}){r}")
    print(f"  {color}{'─'*54}{r}")
    if d != "WAIT" and entry > 0:
        print(f"  {'진입가':<10} {b}${entry:>10,.0f}{r}")
        print(f"  {'손절가':<10} {C['red']}${sl:>10,.0f}{r}   {plan.get('sl_reason','')}")
        print(f"  {'1차 익절':<10} {C['green']}${tp1:>10,.0f}{r}  RR {calc_rr(entry,sl,tp1)}")
        print(f"  {'2차 익절':<10} {C['green']}${tp2:>10,.0f}{r}  RR {calc_rr(entry,sl,tp2)}")
        print(f"  {'3차 익절':<10} {C['green']}{b}${tp3:>10,.0f}{r}  RR {calc_rr(entry,sl,tp3)}")
        if plan.get("liquidation_price"):
            print(f"  {'청산가':<10} {C['red']}{b}${plan['liquidation_price']:>10,.0f}{r}")
        if plan.get("max_loss_usdt"):
            print(f"  {'최대손실':<10} {C['red']}${plan['max_loss_usdt']:>10,.0f} USDT{r}")
        if plan.get("position_btc"):
            print(f"  {'포지션':<10} {plan['position_btc']:.3f} BTC")
        if plan.get("split_entry"):
            print(f"  {'분할진입':<10} {plan['split_entry']}")
    print(f"  {C['gray']}진입근거: {plan.get('entry_reason','')}{r}")
    print(f"  {C['gray']}무효화:   {plan.get('invalidation','')}{r}")
    if plan.get("key_levels"):
        print(f"  {C['gray']}주요레벨: {' | '.join(str(x) for x in plan['key_levels'])}{r}")
    print(f"  {C['gray']}요약: {plan.get('summary','')}{r}")


def print_results(state: State):
    b, r = C["bold"], C["reset"]
    price = state["price"]
    fp    = state.get("final_plan", {})

    print(f"\n{'='*60}")
    print(f"{b}  BTC 선물 분석 결과  {state['timestamp']}{r}")
    print(f"{'='*60}")
    print(f"  현재가: {b}${price:,.2f}{r}  |  24H: {state['change_24h']:+.2f}%")
    print(f"  고가: ${state['high_24h']:,.2f}  |  저가: ${state['low_24h']:,.2f}")
    print(f"  펀딩비: {state['funding_rate']:+.4f}%  |  OI: {state['open_interest']:,.0f} BTC")
    print(f"  공탐지: {state['fear_greed']} ({state['fear_greed_label']})  |  도미넌스: {state['dominance']:.1f}%")
    print(f"  자본금: ${state['capital']:,.0f}  |  레버리지: {state['leverage']}x  |  TF: {state['timeframe']}")

    print(f"\n{b}{'─'*60}")
    print(f"  3개 관점 분석 결과")
    print(f"{'─'*60}{r}")
    print_plan_card("LLM-1  SMC 관점",      C["blue"],   state.get("smc_plan",  {}), price)
    print_plan_card("LLM-2  TA 관점",       C["green"],  state.get("ta_plan",   {}), price)
    print_plan_card("LLM-3  리스크 관점",   C["yellow"], state.get("risk_plan", {}), price)

    if not fp:
        return

    d  = fp.get("direction","WAIT")
    dc = C["green"] if d=="LONG" else C["red"] if d=="SHORT" else C["yellow"]
    entry = fp.get("entry", 0)
    sl    = fp.get("sl",    0)
    tp1   = fp.get("tp1",   0)
    tp2   = fp.get("tp2",   0)
    tp3   = fp.get("tp3",   0)

    print(f"\n{'='*60}")
    print(f"{C['purple']}{b}  최종 종합 플랜 (Aggregator){r}")
    print(f"{'='*60}")
    print(f"\n  방향: {dc}{b}{d}{r}   신뢰도: {b}{fp.get('confidence')}{r}")
    print(f"  합의: {fp.get('consensus','')}")
    print(f"  포지션: {fp.get('position_size','')}")
    print(f"  분할진입: {fp.get('split_entry','')}")

    if d != "WAIT" and entry > 0:
        print(f"\n  {'항목':<12}  {'가격':>12}   RR")
        print(f"  {'─'*44}")
        print(f"  {'진입가':<12}  {C['blue']}{b}${entry:>10,.0f}{r}")
        print(f"  {'손절가':<12}  {C['red']}${sl:>10,.0f}{r}   {fp.get('sl_reason','')}")
        print(f"  {'1차 익절':<12}  {C['green']}${tp1:>10,.0f}{r}   {calc_rr(entry,sl,tp1)}")
        print(f"  {'2차 익절':<12}  {C['green']}${tp2:>10,.0f}{r}   {calc_rr(entry,sl,tp2)}")
        print(f"  {'3차 익절':<12}  {C['green']}{b}${tp3:>10,.0f}{r}   {calc_rr(entry,sl,tp3)}")
        if fp.get("liquidation"):
            print(f"  {'청산가':<12}  {C['red']}{b}${fp['liquidation']:>10,.0f}{r}")

    print(f"\n  {b}각 관점 요약:{r}")
    print(f"  SMC    : {fp.get('smc_verdict','')}")
    print(f"  TA     : {fp.get('ta_verdict','')}")
    print(f"  Risk   : {fp.get('risk_verdict','')}")
    print(f"\n  {b}진입 근거:{r}  {fp.get('entry_reason','')}")
    print(f"  {b}무효화 조건:{r} {C['yellow']}{fp.get('invalidation','')}{r}")
    if fp.get("warnings"):
        print(f"\n  {C['yellow']}{b}주의사항:{r}")
        for w in fp["warnings"]:
            print(f"  {C['yellow']}  ⚠ {w}{r}")
    print(f"\n  {b}즉시 실행 권고:{r}")
    print(f"  {C['cyan']}{fp.get('action','')}{r}")
    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────
# 대화형 입력
# ─────────────────────────────────────────────
def get_user_inputs() -> dict:
    b, r, cy = C["bold"], C["reset"], C["cyan"]
    print(f"\n{cy}{b}{'─'*60}")
    print(f"  추가 설정 입력  (엔터 = 기본값)")
    print(f"{'─'*60}{r}\n")

    def gf(prompt, default):
        try:
            v = input(f"  {prompt} [{default}]: ").strip()
            return float(v) if v else default
        except ValueError:
            return default

    def gi(prompt, default):
        try:
            v = input(f"  {prompt} [{default}]: ").strip()
            return int(v) if v else default
        except ValueError:
            return default

    capital   = gf("분석 자본금 (USDT)", 10_000)
    leverage  = gi("레버리지 배수 (x)",  5)

    print(f"  타임프레임 선택:")
    print(f"  1.1m  2.5m  3.15m  4.1H  5.4H  6.1D")
    tf_map = {"1":"1분봉","2":"5분봉","3":"15분봉","4":"1시간봉","5":"4시간봉","6":"일봉"}
    tf_in  = input(f"  선택 (1~6) [4]: ").strip()
    timeframe = tf_map.get(tf_in, "1시간봉")

    print(f"\n  추가 시장 정보 (FOMC 일정, 주요 뉴스 등, 엔터 = 생략)")
    extra = input("  입력: ").strip()

    return {
        "capital":   capital,
        "leverage":  leverage,
        "timeframe": timeframe,
        "extra_info": extra or "없음",
    }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def choose_model():
    print(f"\n{C['cyan']}{C['bold']}{'─'*60}")
    print(f"  언어모델 선택")
    print(f"{'─'*60}{C['reset']}\n")
    print(f"  1. Gemini (gemini-2.5-flash)")
    print(f"  2. Groq (qwen/qwen3-32b)")
    m_in = input(f"  선택 (1~2) [1]: ").strip()
    
    global SELECTED_MODEL
    if m_in == "2":
        if groq_llm:
            SELECTED_MODEL = "groq"
        else:
            print(f"  {C['yellow']}⚠ GROQ_API_KEY 설정 실패 혹은 누락으로 인해 기본값(Gemini)로 임의 진행합니다.{C['reset']}")
            SELECTED_MODEL = "gemini"
    else:
        SELECTED_MODEL = "gemini"

def main():
    choose_model()
    print_workflow_diagram()

    # 1. 실시간 데이터 수집
    market_data = fetch_market_data()
    market_data = manual_input_price(market_data)
    print(f"\n  {C['bold']}수집 완료 ─ BTC 선물가: ${market_data['price']:,.2f}{C['reset']}")

    # 2. 사용자 입력
    user_inputs = get_user_inputs()

    # 3. State 구성
    initial_state: State = {
        **market_data,
        **user_inputs,
        "smc_plan":   {},
        "ta_plan":    {},
        "risk_plan":  {},
        "final_plan": {},
    }

    # 4. 확인
    print(f"\n  {C['cyan']}{'─'*60}")
    print(f"  자본: ${initial_state['capital']:,.0f} USDT  |  레버리지: {initial_state['leverage']}x"
          f"  |  TF: {initial_state['timeframe']}")
    print(f"{'─'*60}{C['reset']}")
    ok = input("  분석 시작? (엔터/y = 시작, n = 종료): ").strip().lower()
    if ok == "n":
        return

    # 5. LangGraph 실행
    workflow = build_workflow()
    chain    = workflow.compile()

    print(f"\n{C['bold']}[WORKFLOW] START → 3개 LLM 병렬 실행{C['reset']}")
    t0 = time.time()

    try:
        final_state = chain.invoke(initial_state)
    except Exception as e:
        print(f"\n{C['red']}[ERROR] 워크플로우 실패: {e}{C['reset']}")
        print(f"{C['yellow']}팁: DEBUG_LLM = True 설정 후 재실행하면 LLM 원본 응답 확인 가능{C['reset']}")
        raise

    elapsed = time.time() - t0
    print(f"\n{C['gray']}소요 시간: {elapsed:.1f}초{C['reset']}")

    # 6. 결과 출력
    print_results(final_state)


if __name__ == "__main__":
    main()