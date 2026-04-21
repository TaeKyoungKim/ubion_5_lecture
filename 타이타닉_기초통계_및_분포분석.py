import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
try:
    df = pd.read_csv('data/titanic.csv')
    print("데이터 로드 성공!")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")

# 2. 분석용 수치형 컬럼 정의
# 데이터 타입이 int64, float64인 컬럼 선택 후, 식별자인 PassengerId는 제외
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'PassengerId' in numeric_cols:
    numeric_cols.remove('PassengerId')

# 3. 기술통계량(Descriptive Statistics) 계산
# 평균, 중앙값, 표준편차, 사분위수 등을 한눈에 확인
desc_stats = df[numeric_cols].describe().T
desc_stats['skewness'] = df[numeric_cols].skew() # 왜도 추가 (분포의 비대칭 정도)
desc_stats['kurtosis'] = df[numeric_cols].kurt() # 첨도 추가 (분포의 뾰족한 정도)

print("\n--- 수치형 변수 기술통계량 ---")
print(desc_stats)

# 4. 시각화 (Univariate Visualization)
# 각 변수별로 히스토그램(분포)과 박스플롯(이상치)을 나란히 배치
def plot_univariate_analysis(data, columns):
    num_cols = len(columns)
    # 한 컬럼당 2개의 그래프 생성. 세로 길이를 (4->5)로 늘려 상하 여유 공간 확보
    fig, axes = plt.subplots(num_cols, 2, figsize=(14, 5 * num_cols))
    
    # 폰트 및 스타일 설정
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(columns):
        # (1) 히스토그램 & KDE (Distribution)
        sns.histplot(data[col], kde=True, ax=axes[i, 0], color='skyblue')
        # pad를 주어 제목과 그래프 간의 여백 확보
        axes[i, 0].set_title(f'Distribution of {col}', fontsize=14, pad=15)
        axes[i, 0].set_xlabel('')
        
        # (2) 박스플롯 (Outliers)
        sns.boxplot(x=data[col], ax=axes[i, 1], color='salmon')
        axes[i, 1].set_title(f'Boxplot of {col}', fontsize=14, pad=15)
        axes[i, 1].set_xlabel('')

    # 서브플롯 간의 상/하(h_pad), 좌/우(w_pad) 여백을 명시적으로 넓게 설정
    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    plt.show()

# 5. SibSp(형제자매/배우자 수) 특별 분석 (왜도 및 첨도 확인)
if 'SibSp' in df.columns:
    print("\n" + "="*50)
    print("--- 5. SibSp(형제자매/배우자 수) 왜도와 첨도 극단성 분석 ---")
    
    sibsp_skew = df['SibSp'].skew()
    sibsp_kurt = df['SibSp'].kurt()
    
    print(f"▶ SibSp 왜도(Skewness): {sibsp_skew:.4f} (양수면 오른쪽 꼬리가 긴 분포)")
    print(f"▶ SibSp 첨도(Kurtosis): {sibsp_kurt:.4f} (정규분포 0보다 크면 뾰족하고 양쪽 꼬리가 두꺼움)\n")
    
    reason = """[정규분포를 따르지 않는 이유 해석]
1. 왜도 관점: 대부분의 승객이 혼자(0) 탔거나 부부/형제 1명(1)과 동반했습니다. 
   따라서 데이터가 0과 1에 극단적으로 치우쳐 있고, 2 이상의 대가족 데이터는 갈수록 희박하게 존재해 그래프가 오른쪽으로 꼬리가 길어지는(Positive Skew) 형태가 됩니다.
2. 첨도 관점: 데이터가 한쪽(0, 1)에 무서울 정도로 솟아있으며(뾰족함), 소수의 대가족(8명 등) 이상치(Outlier)로 인해 두꺼운 꼬리가 생깁니다.
이처럼 데이터 본연의 특성(극소수의 대가족 탑승) 자체가 자연스러운 대칭형 정규분포(종 모양)를 띄기 불가능한 구조입니다."""
    
    print(reason)
    print("="*50 + "\n")

# 시각화 실행 (경고 등을 다 보고 그래프를 띄우기 위해 마지막에 실행)
plot_univariate_analysis(df, numeric_cols)