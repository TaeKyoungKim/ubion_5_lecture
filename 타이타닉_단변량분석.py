import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
df = pd.read_csv('data/titanic.csv')

# 시각화 스타일 설정
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- (1) 성별(Sex) vs 생존(Survived) : 범주형 vs 범주형 ---
# 성별에 따른 생존 확률(평균)을 막대그래프로 시각화
sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0], palette='pastel')
axes[0].set_title('Survival Rate by Sex', fontsize=14)
axes[0].set_ylabel('Survival Probability')

# --- (2) 정박지(Embarked) vs 생존(Survived) : 범주형 vs 범주형 ---
# 승선 항구에 따른 생존 확률 시각화
sns.barplot(x='Embarked', y='Survived', data=df, ax=axes[1], palette='muted')
axes[1].set_title('Survival Rate by Embarked', fontsize=14)
axes[1].set_ylabel('Survival Probability')

# --- (3) 요금(Fare) vs 생존(Survived) : 수치형 vs 범주형 ---
# 생존 여부에 따른 요금의 분포 차이를 박스플롯으로 확인
# (Fare는 이상치가 크므로 y축 범위를 200으로 제한하여 가독성 확보)
sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[2], palette='Set2')
axes[2].set_ylim(0, 200) 
axes[2].set_title('Fare Distribution by Survival', fontsize=14)

plt.tight_layout()
plt.show()

# --- 통계 수치 확인 ---
print("1. 성별 생존율:\n", df.groupby('Sex')['Survived'].mean())
print("\n2. 정박지별 생존율:\n", df.groupby('Embarked')['Survived'].mean())
print("\n3. 생존 여부별 요금 중앙값:\n", df.groupby('Survived')['Fare'].median())