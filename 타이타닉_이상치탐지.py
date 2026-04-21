import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv("data/titanic.csv")

# ========================================================
# [Part 1] IQR(Interquartile Range) 방식을 이용한 통계적 이상치 탐지
# ========================================================
# * IQR 정의: IQR = Q3 - Q1
# * 이상치 경계 산출: 
#   - Lower Bound = Q1 - 1.5 * IQR
#   - Upper Bound = Q3 + 1.5 * IQR

print("\n=== 1. Fare(요금) 변수의 이상치 구간 및 개수 계산 ===")

# 대소문자를 구분하여 'Fare' 컬럼 사용
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"▶ Q1(25%): {Q1:.2f}, Q3(75%): {Q3:.2f}")
print(f"▶ IQR 값: {IQR:.2f}")
print(f"▶ 정상 범위: {lower_limit:.2f} ~ {upper_limit:.2f}")

# 위에서 구한 정상 범위를 수학적으로 벗어난(크거나 작은) 데이터를 아웃라이어로 간주
outliers = df[(df['Fare'] < lower_limit) | (df['Fare'] > upper_limit)]
print(f"▶ Fare(요금) 통계적 이상치 총 개수: {len(outliers)}개")

# ========================================================
# [Part 2] 수치형 변수들의 Boxplot 시각화 (눈으로 보는 이상치)
# ========================================================
# 박스 위, 아래로 길게 찍히는 점들이 통계적으로 구한 이상치(outliers)에 해당합니다.

plt.figure(figsize=(12, 6))

# (1) 요금(Fare) 박스플롯 시각화
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Fare'], color='lightgreen')
plt.title('Fare Boxplot (Check Outliers)', fontsize=14, pad=10)

# (2) 나이(Age) 박스플롯 시각화
# 나이 컬럼은 결측치(NaN)가 포함되어 있을 수 있으므로 dropna()로 제거 후 시각화합니다.
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Age'].dropna(), color='salmon')
plt.title('Age Boxplot (Check Outliers)', fontsize=14, pad=10)

plt.tight_layout()
plt.show()
