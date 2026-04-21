import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:
    from ctgan import TVAE
except ImportError:
    print("ctgan 라이브러리가 설치되어 있지 않습니다. 터미널(프롬프트) 창에서 'uv add ctgan'을 통해 설치해 주세요.")
    import sys
    sys.exit(1)

# 이전 코드 파일에 정의된 함수를 깔끔하게 임포트하여 그대로 사용합니다.
from cGAN_imbalanced_data import create_dummy_imbalanced_data

warnings.filterwarnings('ignore')

def main():
    # 1. 임의의 불균형 데이터 생성 (모듈 임포트 활용)
    print("\n==== 1. 불균형 데이터 준비 (cGAN_imbalanced_data.py 참조) ====")
    df = create_dummy_imbalanced_data(n_samples=1000, ratio=0.1)
    
    # 소수 데이터 분리
    minority_data = df[df['Label'] == 1].copy()
    majority_data = df[df['Label'] == 0].copy()
    
    # 2. TVAE 모델 컴파일 및 학습
    print("\n==== 2. TVAE 모델 학습 시작 ====")
    print("이 과정은 수 분 정도 소요될 수 있습니다...")
    
    discrete_columns = ['Job_Type', 'Label']
    
    # 구조가 동일한 ctgan 패키지의 TVAE 클래스를 로드합니다.
    tvae = TVAE(epochs=100) 
    tvae.fit(minority_data, discrete_columns)
    print("TVAE 학습 완료!")
    
    # 3. 소수 클래스 증강 
    print("\n==== 3. 소수 데이터 증강 ====")
    num_to_generate = len(majority_data) - len(minority_data)
    print(f"생성할 소수 클래스(Synthetic) 데이터 수: {num_to_generate}개")
    
    synthetic_minority_data = tvae.sample(num_to_generate)
    
    # 4. 시각화 처리
    print("\n==== 4. 분포 시각화 및 검증 ====")
    df_vis = df.copy()
    df_vis['Type'] = df_vis['Label'].map({0: 'Majority Original(0)', 1: 'Minority Original(1)'})
    
    synthetic_vis = synthetic_minority_data.copy()
    synthetic_vis['Type'] = 'Minority Synthetic(1_Gen)'
    
    combined_vis = pd.concat([df_vis, synthetic_vis])
    
    # CTGAN 결과와 동일하게 그리기 우선순위 지정
    combined_vis['Order'] = combined_vis['Type'].map({'Majority Original(0)': 1, 'Minority Synthetic(1_Gen)': 2, 'Minority Original(1)': 3})
    combined_vis = combined_vis.sort_values('Order')
    
    plt.figure(figsize=(14, 6))
    
    # 4-1. 원본 불균형 데이터 (좌측)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_vis, x='Age', y='Income', hue='Type', 
                    hue_order=['Majority Original(0)', 'Minority Original(1)'],
                    palette={'Majority Original(0)': '#f5a3a3', 'Minority Original(1)': 'blue'}, 
                    alpha=0.7, edgecolor='w', s=50)
    plt.title("Original Imbalanced Data")
    plt.xlabel("Age (Continuous)")
    plt.ylabel("Income (Continuous)")
    plt.legend()
    
    # 4-2. TVAE 증강 결과 (우측)
    plt.subplot(1, 2, 2)
    # CTGAN과의 비교를 위해 이번엔 생성 데이터를 오렌지 색으로 설정했습니다.
    sns.scatterplot(data=combined_vis, x='Age', y='Income', hue='Type', 
                    hue_order=['Majority Original(0)', 'Minority Original(1)', 'Minority Synthetic(1_Gen)'],
                    palette={'Majority Original(0)': '#f5a3a3', 
                             'Minority Original(1)': 'blue', 
                             'Minority Synthetic(1_Gen)': '#ff9800'}, # 주황색 
                    alpha=0.7, edgecolor='w', s=40)
    plt.title("Augmented Data with TVAE")
    plt.xlabel("Age (Continuous)")
    plt.ylabel("Income (Continuous)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("tvae_augmentation_result.png", dpi=300)
    print("=> 'tvae_augmentation_result.png' 파일 저장 완료!")
    
    # 4-3. 범주형(디스크리트) 변수 빈도 비교
    plt.figure(figsize=(7, 5))
    sns.countplot(data=combined_vis[combined_vis['Label'] == 1], x='Job_Type', hue='Type',
                  palette={'Minority Original(1)': 'blue', 'Minority Synthetic(1_Gen)': '#ff9800'})
    plt.title("Job_Type Distribution (Original vs TVAE Synthetic)")
    plt.savefig("tvae_job_distribution.png", dpi=300)
    print("=> 'tvae_job_distribution.png' 저장 완료!")

if __name__ == "__main__":
    main()
