import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("data/titanic.csv")

# print(df.columns)

# 1. 생존 분포 도수분포표 확인 (간단하게 value_counts 연속 사용)
# Titanic 데이터셋의 컬럼명은 대소문자를 구분하므로 'Survived'로 변경합니다.
counts = df['Sex'].value_counts()
print("--- 남자(male) 및 여자(female) 인원수 ---")
print(counts)

# 2. Matplotlib 기반의 Pandas 내장 플롯 시각화 (파이 차트)
# 복잡한 설정 없이 .plot.pie() 하나로 비율까지 간결하게 그릴 수 있습니다.
counts.plot.pie(
    labels=['male', 'female'], # 라벨 지정 (0이 많은 경우가 보통이므로 인덱스 0부터 작성)
    autopct='%.1f%%',                     # 소수점 1자리 퍼센트 표시
    colors=['#ff9999', '#66b3ff'],        # 색상 지정
    startangle=90,                        # 12시 방향에서 시작
    title='Sex Distribution'                 # 그래프 제목
)

plt.ylabel('') # 기본으로 생성되는 불필요한 y축 텍스트 제거
plt.show()
