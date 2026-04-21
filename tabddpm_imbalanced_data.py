import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

try:
    from synthcity.plugins import Plugins
except ImportError:
    print("TabDDPM 사용을 위한 synthcity 패키지가 설치되어 있지 않습니다.")
    print("터미널(프롬프트) 창에서 'uv add synthcity' 또는 'pip install synthcity'를 통해 설치해 주세요.")
    sys.exit(1)

# 이전 코드 파일에 정의된 함수를 깔끔하게 임포트하여 그대로 사용합니다.
from cGAN_imbalanced_data import create_dummy_imbalanced_data

warnings.filterwarnings('ignore')

def main():
    # 1. 임의의 불균형 데이터 생성 (기존 함수 재사용)
    print("\n==== 1. 불균형 데이터 준비 (cGAN_imbalanced_data.py 참조) ====")
    df = create_dummy_imbalanced_data(n_samples=1000, ratio=0.1)
    
    # 소수 데이터만 분리
    minority_data = df[df['Label'] == 1].copy()
    majority_data = df[df['Label'] == 0].copy()
    
    # 2. TabDDPM 모델 파라미터 초기화 및 학습
    print("\n==== 2. TabDDPM (Diffusion) 모델 학습 시작 ====")
    print("주의: 디퓨전(Diffusion) 모델은 타 모델보다 학습과 생성에 더 오랜 시간과 연산 자원이 필요합니다...")
    
    # Synthcity 패키지의 플러그인 모듈에서 TabDDPM (ddpm) 객체를 로드합니다.
    # 파라미터(n_iter)는 시간상 테스트 목적으로 500 내외 수준으로 잡습니다.
    tabddpm = Plugins().get("ddpm", n_iter=500) 
    
    # 학습(fit) 진행
    tabddpm.fit(minority_data)
    print("TabDDPM 학습 완료!")
    
    # 3. 소수 클래스 증강 
    print("\n==== 3. 소수 데이터 증강 ====")
    num_to_generate = len(majority_data) - len(minority_data)
    print(f"역확산 프로세스로 생성할 데이터 수: {num_to_generate}개")
    
    # 디퓨전 프로세스로 생성된 데이터를 DataFrame으로 변환
    synthetic_minority_data = tabddpm.generate(count=num_to_generate).dataframe()
    
    # 4. 시각화 처리
    print("\n==== 4. 분포 시각화 및 검증 ====")
    df_vis = df.copy()
    df_vis['Type'] = df_vis['Label'].map({0: 'Majority Original(0)', 1: 'Minority Original(1)'})
    
    synthetic_vis = synthetic_minority_data.copy()
    synthetic_vis['Type'] = 'Minority Synthetic(1_Gen)'
    
    combined_vis = pd.concat([df_vis, synthetic_vis])
    
    # 그리기 우선순위 지정 (다수 데이터 -> 디퓨전 모델 생성 데이터 -> 소수 원본 데이터)
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
    
    # 4-2. TabDDPM 증강 결과 (우측)
    plt.subplot(1, 2, 2)
    # 구분을 명확히 하기 위해 TabDDPM 증강 데이터 색상은 '보라색(Deep Purple)' 계통으로 지정했습니다.
    sns.scatterplot(data=combined_vis, x='Age', y='Income', hue='Type', 
                    hue_order=['Majority Original(0)', 'Minority Original(1)', 'Minority Synthetic(1_Gen)'],
                    palette={'Majority Original(0)': '#f5a3a3', 
                             'Minority Original(1)': 'blue', 
                             'Minority Synthetic(1_Gen)': '#9c27b0'}, # 딥퍼플
                    alpha=0.7, edgecolor='w', s=40)
    plt.title("Augmented Data with TabDDPM (Diffusion)")
    plt.xlabel("Age (Continuous)")
    plt.ylabel("Income (Continuous)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("tabddpm_augmentation_result.png", dpi=300)
    print("=> 'tabddpm_augmentation_result.png' 저장 완료!")
    
    # 4-3. 범주형(디스크리트) 변수 빈도 비교
    plt.figure(figsize=(7, 5))
    sns.countplot(data=combined_vis[combined_vis['Label'] == 1], x='Job_Type', hue='Type',
                  palette={'Minority Original(1)': 'blue', 'Minority Synthetic(1_Gen)': '#9c27b0'})
    plt.title("Job_Type Distribution (Original vs DDPM Synthetic)")
    plt.savefig("tabddpm_job_distribution.png", dpi=300)
    print("=> 'tabddpm_job_distribution.png' 저장 완료!")

if __name__ == "__main__":
    main()
