import pandas as pd
import plotly.express as px

df = pd.read_csv("data/titanic.csv")

print("=== 1. pandas의 대표적인 결측치 조회 함수 ===")

# (1) df.isna() / df.isnull() : 데이터가 결측치인지 여부를 True/False로 반환
print("\n[1] df.isna().head(3) - 결측치면 True")
print(df.isnull().head(3))

# (2) df.notna() / df.notnull() : 데이터가 정상(Not NaN)인지 여부를 True/False로 반환
print("\n[2] df.notna().head(3) - 정상이면 True")
print(df.notna().head(3))

# (3) df.isna().any() : 각 컬럼(열)에 결측치가 '하나라도' 있는지 확인
print("\n[3] df.isna().any() - 컬럼별 결측치 존재 여부")
print(df.isna().any())

# (4) df.isna().sum() : 각 컬럼별 결측치 개수 파악
print("\n[4] df.isna().sum() - 컬럼별 결측치 총 개수")
print(df.isna().sum())

# (5) df.isna().sum().sum() : 데이터프레임 전체의 모든 결측치 총 개수
print("\n[5] df.isna().sum().sum() - 전체 결측치 총합")
print(df.isna().sum().sum())

print("\n=== 2. 위 조회 함수를 활용한 결측치 시각화 (Plotly) ===")

# 시각화를 위해 결측치가 1개 이상 있는 컬럼들만 내림차순으로 정렬
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

# 2. Plotly를 활용한 간략한 결측치 시각화
fig = px.bar(
    x=missing.index, 
    y=missing.values, 
    title="Titanic 데이터 결측치 현황",
    labels={'x': '컬럼명', 'y': '결측치 개수'},
    text_auto=True # 막대 위에 숫자 표시
)

fig.show()
