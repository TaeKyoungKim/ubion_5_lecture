import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings

# 깔끔한 출력을 위한 워닝 무시
warnings.filterwarnings('ignore')

class DBSCANOptimizationManager:
    """
    고성능 Hyperparameter Optimization for DBSCAN Pipeline
    - 데이터 전처리 (StandardScaler 필수)
    - eps (0.1 ~ 1.0) & min_samples (2 ~ 10) Grid Search 수행
    - Silhouette Score 히트맵 분석 및 기하학적 난제(Moons) 해결 성능 시각화
    
    [전문가 지식 가이드: eps 크기에 따른 이상 현상]
    1. eps가 너무 작을 때 (Over-strict):
       데이터 간의 밀집 반경(거리) 조건이 너무 엄격해지므로, 정상적인 포인트마저도 
       이웃을 찾지 못하여 고립됩니다. 이 경우 클러스터가 하나도 형성되지 않고 
       데이터 거의 전부가 흩어진 Noise(노이즈, -1)로 파괴되는 현상이 발생합니다.
    2. eps가 너무 클 때 (Over-relaxed):
       반경 조건이 데이터의 분산 범위보다 느슨해져서, 본래라면 구분되어야 할 멀리 떨어진 
       정보들까지 모두 이웃으로 편입됩니다. 결국 데이터의 선형/비선형적 구조 파악에 
       모두 실패하고 전체 데이터가 1개의 거대한 통짜 군집(Single Cluster)으로 녹아내리게 됩니다.
    """
    def __init__(self):
        self.X_raw = None
        self.y_true = None
        self.X_scaled = None
        self.X_pca = None
        
        self.eps_range = np.round(np.arange(0.1, 1.1, 0.1), 1)
        self.min_samples_range = range(2, 11)
        self.score_matrix = None
        
        self.best_eps = None
        self.best_min_samples = None
        self.best_score = -1
        
        self.best_dbscan = None
        self.labels = None

    def data_preparation(self):
        print("1. Data Preparation: DBSCAN 맞춤형 비선형(Non-Linear) 임의 데이터 생성 중...")
        # DBSCAN의 강점(비선형적 모양 군집화)을 가장 잘 나타낼 수 있는 make_moons 데이터셋 사용
        self.X_raw, self.y_true = make_moons(n_samples=400, noise=0.08, random_state=42)
        
        # DBSCAN은 대표적인 "밀도(거리)" 기반 알고리즘이므로, 피처 간 절대적 스케일 차이가 
        # 밀도 측정에 심각한 왜곡을 주기 때문에 완벽한 표준화를 거쳐주어야 합니다.
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X_raw)
        print("   [완료] 초승달(Moons) 2개가 겹쳐져 있는 복잡한 기하학적 구조 데이터를 생성 후 표준화했습니다.\n")
        
        # 2차원 시각화를 위한 PCA 적용 (원본이 2D이더라도 일관된 Pipeline 구조 유지 위함)
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(self.X_scaled)

    def optimize_hyperparameters(self):
        print("2. Hyperparameter Optimization: (eps, min_samples) 2D Grid Search 탐색 중...")
        # Heatmap을 위한 비어있는 DataFrame 초기화
        self.score_matrix = pd.DataFrame(
            index=self.min_samples_range, 
            columns=self.eps_range, 
            dtype=float
        )
        
        for ep in self.eps_range:
            for ms in self.min_samples_range:
                dbscan = DBSCAN(eps=ep, min_samples=ms)
                temp_labels = dbscan.fit_predict(self.X_scaled)
                
                # 순수하게 식별된 군집(Cluster)의 수를 카운트 (-1 노이즈 레이블은 제외)
                unique_clusters = set(temp_labels) - {-1}
                n_clusters = len(unique_clusters)
                
                # 실루엣 계수는 1개가 아닌 2개 이상의 클러스터가 필요함
                # 클러스터링을 강제로 실패(합치거나 쪼갬)한 파라미터 조합은 평가에서 배제함(NaN 출력)
                if n_clusters >= 2:
                    score = silhouette_score(self.X_scaled, temp_labels)
                    self.score_matrix.at[ms, ep] = score
                    
                    # 최고 성능 업데이트 구간
                    if score > self.best_score:
                        self.best_score = score
                        self.best_eps = ep
                        self.best_min_samples = ms

        print(f"   [탐색 종료] 최종 산출물: 실루엣 수치 극대화 기준 최적 파라미터 🎯")
        print(f"       => 최적 eps: {self.best_eps} / 최적 min_samples: {self.best_min_samples} (Silhouette Score: {self.best_score:.4f})")
        
    def run_optimal_model(self):
        print("\n3. 산출된 최적 파라미터 (eps, min_samples) 기반 Main DBSCAN 모델링...")
        self.best_dbscan = DBSCAN(eps=self.best_eps, min_samples=self.best_min_samples)
        self.labels = self.best_dbscan.fit_predict(self.X_scaled)
        
        # 모델 그룹화 및 노이즈 비율 산출
        total_points = len(self.labels)
        noise_points = list(self.labels).count(-1)
        noise_ratio = (noise_points / total_points) * 100
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        
        print(f"   [완료] 최적 파라미터 학습 결과, 총 ✨{n_clusters}개✨의 비선형 클러스터가 성공적으로 도출되었습니다.")
        print(f"   [완료] 노이즈(Outlier) 분류: 총 {noise_points}개 데이터 포인트 탐지 (전체의 {noise_ratio:.1f}%)")

    def visualize_and_report(self):
        print("\n4. Visual Analytics: 파라미터 탐색 히트맵 및 해결된 비선형 클러스터 산점도 렌더링 중...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # --- (1) eps, min_samples 조합에 대한 Silhouette Score Heatmap ---
        ax1 = axes[0]
        # NaN 인덱스는 자동으로 공백으로 출력되어 시각적으로 실패 구간을 필터링해 반영함
        sns.heatmap(self.score_matrix, annot=True, fmt=".3f", cmap='rocket_r', 
                    ax=ax1, cbar_kws={'label': 'Silhouette Score'}, annot_kws={"size": 10})
        ax1.set_title("Grid Search: Silhouette Score Density Map", fontsize=14)
        ax1.set_xlabel("EPS (Radius of neighborhoods)", fontsize=12)
        ax1.set_ylabel("Min_Samples (Density threshold)", fontsize=12)
        ax1.invert_yaxis()

        # --- (2) DBSCAN Clustering PCA Scatter Plot ---
        ax2 = axes[1]
        unique_labels = set(self.labels)
        
        for k in unique_labels:
            class_members = (self.labels == k)
            
            # 노이즈(-1) 마스터링 처리: 검정색 거대 X마크로 극명하게 식별
            if k == -1:
                ax2.scatter(
                    self.X_pca[class_members, 0], self.X_pca[class_members, 1],
                    color='black', marker='X', s=130, alpha=0.9, label='Noise Outliers (-1)', zorder=10
                )
            else:
                ax2.scatter(
                    self.X_pca[class_members, 0], self.X_pca[class_members, 1],
                    marker='o', s=80, alpha=0.9, edgecolor='k', label=f'Density Cluster {k}'
                )

        ax2.set_xlabel('Principal Component 1 (PC1)', fontsize=12)
        ax2.set_ylabel('Principal Component 2 (PC2)', fontsize=12)
        ax2.set_title(f'DBSCAN Non-linear Clustering Triumph\n(Optimal eps={self.best_eps}, min_samples={self.best_min_samples})', fontsize=14)
        ax2.legend(loc='best', fontsize=11, title="Separated Groups")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # 환경 저장을 위한 동적 패스
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(current_dir, "dbscan_clustering_optimization.png")
        plt.savefig(img_path, dpi=300)
        print(f"   [완료] 시각화 이미지 듀얼 저장 경로: {img_path}")

    def print_expert_review(self):
        opinion = f"""
================================================================================
[전문가 리뷰: DBSCAN의 비선형적(Non-linear) 기하학 구조 해결의 압도적 우위]

1. K-Means의 한계 (원형 구조의 함정)
K-Means나 가우시안분포(GMM) 등은 오직 '데이터의 중심점과의 반경(거리/확률)'만을 
기준으로 삼습니다. 때문에 이번에 인위적으로 생성한 '두 겹의 초승달(Moons)' 모양처럼 
복잡하게 구불구불 이어지거나 서로 맞물려있는 기하학적 형태의 난해한 분포를 마주하면 
절대로 군집을 찾지 못하고 한가운데를 가로로 잘라먹는 대형 오분류를 저지릅니다.

2. 밀도 전파(Density Reachability)를 바탕으로 한 기하학적 해결 능력
반면에 적합한 하이퍼파라미터(eps={self.best_eps})가 장착된 DBSCAN 알고리즘은,
데이터가 밀집된 지점에서부터 촘촘하게 다음 이웃으로 확산해나가는 '사슬 연결망 
(Core-Border Network)' 방식을 지니기 때문에, 아무리 뱀이나 초승달처럼 비정형하게 
휘어져 있는 데이터일지라도 완벽하게 그 곡률을 쫓아가며 구조를 짚어내는 쾌거를 거둡니다.

3. 노이즈(Outliers)의 완벽한 솎아냄
위 우측 그래프의 거대한 ❌ 마크를 유의 깊게 관찰해 보십시오.
어느 군집의 밀도 경로 반경에도 닿지 못한 채 궤도 바깥으로 튕겨져 나간, 학습 분포를 
심각히 훼방놓는 악성 데이터들을 깔끔하게 노이즈(-1)로 격리시켜 본 모델의 의사결정을 
오염시키지 못하도록 철저히 방어한 특유의 Noise Filtering 메커니즘을 증명하고 있습니다.
================================================================================
"""
        print(opinion)

if __name__ == "__main__":
    task = DBSCANOptimizationManager()
    
    task.data_preparation()
    task.optimize_hyperparameters()
    task.run_optimal_model()
    task.visualize_and_report()
    task.print_expert_review()
