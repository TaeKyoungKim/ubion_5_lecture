import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("data/titanic.csv")

# 1. 성별(Sex)과 객실 등급(Pclass)에 따른 생존율 피벗 테이블 생성
# Titanic 원본 데이터셋의 컬럼명 대소문자(Survived, Sex, Pclass)에 맞춰 수정했습니다.
pivot_table = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')

print("=== 성별 및 객실 등급별 생존율 ===")
# 파이썬 스크립트(`.py`) 환경에서는 주피터 노트북의 display() 대신 print()를 사용합니다.
print(pivot_table)

# 2. 시각화: 히트맵(Heatmap)을 통해 생존율 농도 차이 확인
plt.figure(figsize=(10, 6))

# annot=True: 히트맵 칸 안에 실제 수치 표시
# cmap='RdYlBu_r': 빨간색~노란색~파란색으로 이어지는 색상 테마 (생존율이 높을수록 파란색)
# fmt='.3f': 소수점 아래 3자리까지 출력
sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', fmt='.3f')

plt.title('Survival Rate by Sex and Pclass', fontsize=16, pad=15)
plt.show()
