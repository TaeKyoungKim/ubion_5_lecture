import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import warnings

# 깔끔한 출력을 위한 워닝 무시
warnings.filterwarnings('ignore')

class GMMOptimizationTask:
    """
    고성능 Hyperparameter Optimization for Gaussian Mixture Model Pipeline
    - StandardScaler 적용 데이터 표준화
    - n_components (1~10) Grid Search (BIC / AIC 계수 평가지표 탐색)
    - K-Means와 달리 'Soft Clustering(확률적 할당)'을 지원
    - PCA 시각화 및 GMM Ellipsoids(확률 타원) 생성 모델링
    """
    def __init__(self):
        self.X_raw = None
        self.y_true = None
        self.X_scaled = None
        self.X_pca = None
        
        # n_components 탐색 범위 설정
        self.k_range = range(1, 11)
        self.aic_scores = []
        self.bic_scores = []
        
        self.optimal_k = None
        self.best_gmm = None
        self.labels = None

    def data_preparation(self):
        print("1. Data Preparation: 데이터 로드 및 전처리 시작...")
        iris = load_iris()
        self.X_raw = iris.data
        self.y_true = iris.target
        
        # GMM의 확률 분포 및 가능도(Likelihood) 추정 성능을 높이기 위해 특성 스케일을 정수로 표준화
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X_raw)
        print("   [완료] StandardScaler를 적용하여 데이터를 표준화했습니다.\n")
        
        # 2차원 투영 및 엘립소이드 렌더링 가시성을 위한 PCA 사전 모델링
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(self.X_scaled)

    def optimize_hyperparameters(self):
        print("2. Hyperparameter Optimization: n_components Grid Search 진행 중 (K= 1 ~ 10)...")
        # GMM은 K-Means의 Hard Clustering에 대비되는 Soft Clustering(데이터가 각 군집에 속할 확률적 수치 제공)을 지원.
        # 따라서 단순 유클리드 거리 지표인 Silhouette 대신 통계적 확률 적합성인 BIC, AIC가 필수적.
        
        for k in self.k_range:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(self.X_scaled)
            
            # BIC (Bayesian Information Criterion) 및 AIC (Akaike Information Criterion) 산출
            aic = gmm.aic(self.X_scaled)
            bic = gmm.bic(self.X_scaled)
            
            self.aic_scores.append(aic)
            self.bic_scores.append(bic)
            print(f"   => Component(K): {k:2d} | AIC Score: {aic:.2f} | BIC Score: {bic:.2f}")

        # 가장 낮은 BIC 수치를 보유한 K를 최상의 모델 성분 갯수로 선정 (모델 과적합에 대한 엄격한 패널티 기준)
        self.optimal_k = self.k_range[np.argmin(self.bic_scores)]
        print(f"\n   🌟 최종 산출물: BIC 수치 최소화 추적에 따른 최적 n_components 🎯 K={self.optimal_k}")
        
    def run_optimal_model(self):
        print(f"\n3. 선정된 최적 K({self.optimal_k}) 기반 Main GMM 모델링...")
        self.best_gmm = GaussianMixture(n_components=self.optimal_k, covariance_type='full', random_state=42)
        # GMM의 진면목인 확률적 할당의 결과로 가장 가능성(Posterior)이 높은 정수 레이블을 도출시킵니다.
        self.labels = self.best_gmm.fit_predict(self.X_scaled)
        print(f"   [완료] 최적 GMM 학습 및 최종 Soft Clustering 결과 그룹화 도출 완료.")

    def make_ellipses(self, gmm, ax, colors):
        """(Advanced) Scikit-learn 권장 GMM 확률 분포 타원 시각화 헬퍼 함수"""
        for n, color in enumerate(colors):
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
                
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # Convert degree
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            
            # 가우시안 확률 분포의 표준 편차 반경인 Ellipse를 그림
            ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.3)
            ax.add_artist(ell)
            # 타원의 중심점 마킹
            ax.scatter(gmm.means_[n, 0], gmm.means_[n, 1], marker='x', c='red', s=100, linewidth=3, zorder=10)

    def visualize_and_report(self):
        print("\n4. Visual Analytics: BIC/AIC 분석 곡선 그래프 및 GMM 2D Ellipsoids 렌더링 중...")
        
        # 1x2 Subplot 파노라마 구도
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- (1) BIC / AIC 최적화 궤적 Plot ---
        ax1 = axes[0]
        ax1.plot(self.k_range, self.aic_scores, marker='o', label='AIC', linewidth=2, color='royalblue')
        ax1.plot(self.k_range, self.bic_scores, marker='s', label='BIC', linewidth=2, color='darkorange')
        ax1.axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K ({self.optimal_k})')
        ax1.set_xlabel('Number of Components (K)', fontsize=12)
        ax1.set_ylabel('Information Criterion', fontsize=12)
        ax1.set_title('GMM Model Selection (AIC/BIC Minimization)', fontsize=14)
        ax1.set_xticks(self.k_range)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- (2) GMM Ellipsoids & PCA 투영 산점도 ---
        ax2 = axes[1]
        
        # Ellipsoids(타원)를 아름답게 2차원에 시각화하기 위해 PCA 기반 데이터로 임시 2D GMM을 학습시킵니다.
        # 본래 실전에서 타원은 n차원에 형성되지만, 차원 축소된 시각화 공간에 매끄럽게 호환되도록 처리하는 고급 기법입니다.
        gmm_2d = GaussianMixture(n_components=self.optimal_k, covariance_type='full', random_state=42)
        gmm_2d.fit(self.X_pca)
        
        # 군집 구분을 위한 다이나믹 컬러맵 할당
        cmap = plt.get_cmap('viridis')
        colors_list = [cmap(i) for i in np.linspace(0, 0.9, self.optimal_k)]
        
        for k in range(self.optimal_k):
            class_members = (self.labels == k)
            ax2.scatter(
                self.X_pca[class_members, 0],
                self.X_pca[class_members, 1],
                color=colors_list[k],
                marker='o',
                edgecolors='k',
                s=60,
                alpha=0.8,
                label=f'Cluster {k}'
            )

        # Scikit-learn 표준 타원(Ellipsoids) 렌더링 호출
        self.make_ellipses(gmm_2d, ax2, colors_list)
        
        ax2.set_xlabel('Principal Component 1 (PC1)', fontsize=12)
        ax2.set_ylabel('Principal Component 2 (PC2)', fontsize=12)
        ax2.set_title(f'GMM Soft Clustering with Ellipsoids (K={self.optimal_k})', fontsize=14)
        ax2.legend(loc='lower right', title='Clusters', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # 현재 스크립트 실행 위치에 이미지 저장
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(current_dir, "gmm_clustering_optimization.png")
        plt.savefig(img_path, dpi=300)
        print(f"   [완료] 시각화 듀얼 이미지(1x2) 저장 결로: {img_path}")

    def print_expert_review(self):
        opinion = f"""
================================================================================
[전문가 리뷰: GMM의 통계적 이점 및 BIC/AIC 평가의 당위성]

1. Soft Clustering (확률적 분배)의 압도적 유연성
기존에 다루었던 K-Means는 모든 데이터를 가장 가까운 클러스터에 무조건 1개씩 박아넣는 
'Hard Clustering / 흑백 논리' 구조입니다. 하지만 수리가 기반되는 GMM 모델은 
'이 데이터 포인트가 0번 그룹일 확률 80%, 1번 그룹일 확률 20%' 처럼 'Soft Clustering'
방식을 따릅니다. 따라서 클러스터 경계선에 걸쳐있는 애매한 데이터들의 확률을 유연하게 
처리할 수 있으며, 원형뿐만 아니라 길쭉한 타원 모양(Elliptical shape)의 굴곡진 데이터 
분포도 매우 유기적으로 감싸 모델링할 수 있는 강점이 있습니다.

2. 왜 실루엣 점수(Silhouette)가 아니라 BIC / AIC 인가?
GMM은 확률 분포가 얼마나 데이터를 잘 설명하는지를 따지는 모델(Generative Model)이기 
때문에, 단순히 데이터 사이트 간의 물리적 거리(유클리드)만 따지는 '실루엣 스코어'와는 
궁합이 매우 나쁩니다. 그보다는 데이터 설명력(Likelihood, 우도)을 극대화하되 군집 개수(K)를 
무분별하게 늘리는 등 변수 삽입이 많아지면 벌점(Penalty)을 주는 통계적 지표인 
'AIC'와 'BIC'가 교과서적 표준입니다.

- 선형 비교시: BIC는 변수(K)가 추가됨에 따라 로그 단위의 페널티를 훨씬 더 강하게 주므로,
  과적합(Overfitting)을 차단하고 보다 Simple한 모델의 구조를 채택하도록 방어막을 칩니다.

금일의 Grid-Search 파이프라인 적용 결과, K=2 또는 K=3 구간 주변에서 정보량이 수직 
낙하하며 'Elbow(엘보우)' 지점이 도출됨과 동시에, 이 중 파라미터 팽창을 구조적으로 
차단한 BIC 계수가 가장 최저점의 바닥을 친 최종 K={self.optimal_k} 가 통계적으로 가장 
강건한 GMM 모델 셋업으로 최종 선정 및 시각화되었습니다.
================================================================================
"""
        print(opinion)

if __name__ == "__main__":
    task = GMMOptimizationTask()
    
    task.data_preparation()
    task.optimize_hyperparameters()
    task.run_optimal_model()
    task.visualize_and_report()
    task.print_expert_review()
