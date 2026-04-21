import plotly.express as px

# 1. 샘플 데이터 준비
# Plotly에서 기본으로 제공하는 식당 팁(tips) 데이터를 불러옵니다.
df = px.data.tips()

# 2. 산점도(Scatter Plot) 생성
# x축은 총 지불 금액(total_bill), y축은 팁(tip)을 나타냅니다.
# 흡연 여부(smoker)에 따라 색상을 다르게 표시하고, 성별(sex)에 따라 그래프를 분리합니다.
fig = px.scatter(
    df, 
    x="total_bill", 
    y="tip", 
    color="smoker", 
    facet_col="sex",
    title="레스토랑 팁 데이터 산점도 (성별 및 흡연 여부 비교)",
    labels={"total_bill": "총 지불 금액 ($)", "tip": "팁 ($)", "smoker": "흡연 여부"}
)

# 3. 그래프 출력
# 스크립트를 실행하면 기본 브라우저가 열리며 그래프가 렌더링됩니다.
fig.show()

# (참고) 인터랙티브한 그래프 상태 그대로 HTML 파일로 저장하고 싶다면 아래 주석을 해제하세요.
# fig.write_html("plotly_python_result.html")
