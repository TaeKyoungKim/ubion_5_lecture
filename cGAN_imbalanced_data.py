import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:
    from ctgan import CTGAN
except ImportError:
    print("ctgan 라이브러리가 설치되어 있지 않습니다. 터미널(프롬프트) 창에서 'uv add ctgan' 또는 'pip install ctgan'을 통해 설치해 주세요.")
    import sys
    sys.exit(1)

warnings.filterwarnings('ignore')

def create_dummy_imbalanced_data(n_samples=1000, ratio=0.1):
    """
    이산적(Discrete) 변수와 연속적(Continuous) 변수가 섞인 가상의 불균형 데이터를 생성합니다.
    - n_samples: 총 데이터 수
    - ratio: 소수 클래스(Label 1)의 비율
    """
    np.random.seed(42)
    minority_n = int(n_samples * ratio)      # 100개
    majority_n = n_samples - minority_n      # 900개
    
    print(f"데이터 생성 중: 총 샘플 수 = {n_samples}, 다수 클래스 = {majority_n}개, 소수 클래스 = {minority_n}개")

    # 1. 다수 클래스 생성 (Label = 0)
    # 연속형(Continuous): Age, Income
    age_maj = np.random.normal(40, 10, majority_n)
    income_maj = np.random.normal(60000, 15000, majority_n)
    # 이산형(Discrete): Job_Type
    job_maj = np.random.choice(['IT', 'Sales', 'HR'], majority_n, p=[0.2, 0.6, 0.2])
    label_maj = np.zeros(majority_n, dtype=int)
    
    # 2. 소수 클래스 생성 (Label = 1)
    age_min = np.random.normal(25, 5, minority_n)
    income_min = np.random.normal(30000, 8000, minority_n)
    job_min = np.random.choice(['IT', 'Sales', 'HR'], minority_n, p=[0.7, 0.2, 0.1])
    label_min = np.ones(minority_n, dtype=int)
    
    df_maj = pd.DataFrame({'Age': age_maj, 'Income': income_maj, 'Job_Type': job_maj, 'Label': label_maj})
    df_min = pd.DataFrame({'Age': age_min, 'Income': income_min, 'Job_Type': job_min, 'Label': label_min})
    
    # 두 데이터를 합치고 무작위로 섞음
    df = pd.concat([df_maj, df_min]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def main():
    # 1. 임의의 불균형 데이터 생성
    print("\n==== 1. 불균형 데이터 준비 ====")
    df = create_dummy_imbalanced_data(n_samples=1000, ratio=0.1)
    print("\n원본 데이터 Label 분포:")
    print(df['Label'].value_counts())
    
    # 소수 데이터만 따로 분리 (증강의 타겟)
    minority_data = df[df['Label'] == 1].copy()
    majority_data = df[df['Label'] == 0].copy()
    
    # 2. CTGAN 모델 컴파일 및 학습
    print("\n==== 2. CTGAN 모델 학습 시작 ====")
    print("이 과정은 수 분 정도 소요될 수 있습니다...")
    
    # CTGAN에 어떤 컬럼이 '이산형(Discrete)'인지 알려주어야 함
    discrete_columns = ['Job_Type', 'Label']
    
    # CTGAN 생성기 초기화 및 소수 클래스에 대한 학습 (epochs 100회)
    # 참고: 전체 데이터가 아닌 소수 클래스(Label=1) 만을 피팅하여 그 고유한 특징을 배움
    ctgan = CTGAN(epochs=100, verbose=True) 
    ctgan.fit(minority_data, discrete_columns)
    print("학습 완료!")
    
    # 3. 소수 클래스 증강 (다수 클래스 개수만큼 복원)
    print("\n==== 3. 소수 데이터 증강 ====")
    num_to_generate = len(majority_data) - len(minority_data)
    print(f"생성할 소수 클래스(Synthetic) 데이터 수: {num_to_generate}개")
    
    synthetic_minority_data = ctgan.sample(num_to_generate)
    augmented_data = pd.concat([df, synthetic_minority_data])
    
    print("\n증강 후 전체 데이터 Label 분포:")
    print(augmented_data['Label'].value_counts())
    
    # 4. 시각화 처리
    print("\n==== 4. 분포 시각화 및 검증 ====")
    
    # 시각화 색상 및 라벨 구분용 타입 컬럼 추가
    df_vis = df.copy()
    df_vis['Type'] = df_vis['Label'].map({0: 'Majority Original(0)', 1: 'Minority Original(1)'})
    
    synthetic_vis = synthetic_minority_data.copy()
    synthetic_vis['Type'] = 'Minority Synthetic(1_Gen)'
    
    combined_vis = pd.concat([df_vis, synthetic_vis])
    
    # 한글 폰트가 깨지지 않도록 영문 레이블 위주로 설정
    plt.figure(figsize=(14, 6))
    
    # 4-1. (좌측) 원본 불균형 데이터 분포
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_vis, x='Age', y='Income', hue='Type', 
                    hue_order=['Majority Original(0)', 'Minority Original(1)'],
                    palette={'Majority Original(0)': '#f5a3a3', 'Minority Original(1)': 'blue'}, 
                    alpha=0.7, edgecolor='w', s=50)
    plt.title("Original Imbalanced Data")
    plt.xlabel("Age (Continuous)")
    plt.ylabel("Income (Continuous)")
    plt.legend()
    
    # 4-2. (우측) CTGAN 증강 후 데이터 분포
    plt.subplot(1, 2, 2)
    # 데이터 포인트가 겹칠 때 시각적으로 구분이 쉽도록 그리는 순서를 조정합니다. (다수 -> 생성 -> 원본 소수)
    combined_vis['Order'] = combined_vis['Type'].map({'Majority Original(0)': 1, 'Minority Synthetic(1_Gen)': 2, 'Minority Original(1)': 3})
    combined_vis = combined_vis.sort_values('Order')
    
    sns.scatterplot(data=combined_vis, x='Age', y='Income', hue='Type', 
                    hue_order=['Majority Original(0)', 'Minority Original(1)', 'Minority Synthetic(1_Gen)'],
                    palette={'Majority Original(0)': '#f5a3a3', 
                             'Minority Original(1)': 'blue', 
                             'Minority Synthetic(1_Gen)': '#4caf50'}, 
                    alpha=0.7, edgecolor='w', s=40)
    
    plt.title("Augmented Data with CTGAN")
    plt.xlabel("Age (Continuous)")
    plt.ylabel("Income (Continuous)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ctgan_augmentation_result.png", dpi=300) # 고해상도 저장
    print("=> 'ctgan_augmentation_result.png' 파일로 스캐터플롯 이미지가 개선되어 생성/저장되었습니다!")
    
    # 4-3. 추가적으로 범주형(디스크리트) 변수 빈도 비교하기
    plt.figure(figsize=(7, 5))
    sns.countplot(data=combined_vis[combined_vis['Label'] == 1], x='Job_Type', hue='Type',
                  palette={'Minority Original(1)': 'blue', 'Minority Synthetic(1_Gen)': 'green'})
    plt.title("Job_Type Distribution (Original vs Synthetic)")
    plt.savefig("ctgan_job_distribution.png", dpi=150)
    print("=> 'ctgan_job_distribution.png' 파일로 Job_Type(이산형) 비교 이미지가 저장되었습니다!")

    # plt.show() # 스크립트 실행 환경에서 창을 띄우려면 주석을 해제할 것

if __name__ == "__main__":
    main()
