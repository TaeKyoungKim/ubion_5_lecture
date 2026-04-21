import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

class PCAKMeansPipeline:
    """
    고성능 PCA Dimensionality Reduction & K-Means Clustering Pipeline
    학습/평가 분리 및 평가지표(Silhouette, Calinski-Harabasz) 최적화 시각화 반영
    """
    def __init__(self):
        self.X_train_raw = None; self.X_test_raw = None
        self.y_train_true = None; self.y_test_true = None
        
        self.X_train_scaled = None; self.X_test_scaled = None
        self.X_train_pca = None; self.X_test_pca = None
        
        self.pca = None; self.kmeans = None
        self.train_labels = None; self.test_labels = None
        
        # 평가 지표 기록용 리스트
        self.k_range = range(2, 9)
        self.sil_scores = []
        self.ch_scores = []

    def run_pipeline(self):
        print("1. Data Pipeline Requirements: 데이터 로드 및 전처리 시작...")
        iris = load_iris()
        
        # Train/Test Split (80% 학습, 20% 평가 데이터로 분리)
        self.X_train_raw, self.X_test_raw, self.y_train_true, self.y_test_true = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        
        # StandardScaler (오직 Train 데이터로만 fit)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train_raw)
        self.X_test_scaled = scaler.transform(self.X_test_raw)
        
        print("2. Step 1: PCA (Feature Extraction)...")
        self.pca = PCA(n_components=2)
        # Train 데이터로 주성분 축을 학습 및 변환
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        var_ratio = self.pca.explained_variance_ratio_
        print(f"   [완료] 총 설명 분산(PC1 + PC2): {(var_ratio[0] + var_ratio[1])*100:.2f}%\n")

        # 최적의 K를 탐색하기 위한 연속적인 평가 지표 산출 (K=2 ~ 8)
        print("3. 클러스터링 평가 지표 탐색 (K=2 ~ 8)...")
        for k in self.k_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
            temp_labels = km.fit_predict(self.X_train_pca)
            self.sil_scores.append(silhouette_score(self.X_train_pca, temp_labels))
            self.ch_scores.append(calinski_harabasz_score(self.X_train_pca, temp_labels))
        print("   [완료] 지표(Silhouette, Calinski-Harabasz) 계산 완료.\n")

        print("4. Step 2: K-Means (Clustering, K=3)...")
        self.kmeans = KMeans(n_clusters=3, init='k-means++', n_init='auto', random_state=42)
        
        # 학습셋 기준점 찾기
        self.train_labels = self.kmeans.fit_predict(self.X_train_pca)
        
        # 평가셋 예측
        self.test_labels = self.kmeans.predict(self.X_test_pca)
        
        # Test 데이터 기준 최종 평가지표 출력
        test_sil = silhouette_score(self.X_test_pca, self.test_labels)
        test_ch = calinski_harabasz_score(self.X_test_pca, self.test_labels)
        
        print("   [완료] 평가셋(Test Dataset) 클러스터 분석 및 평가 완료.")
        print(f"   => 🔮 예측 결과 (Predicted) : {self.test_labels}")
        print(f"   => 🎯 실제 정답 (True)      : {self.y_test_true}")
        print(f"   => 📊 Silhouette Score      : {test_sil:.4f} (1에 가까울수록 응집도/분리도 우수)")
        print(f"   => 📈 Calinski-Harabasz Score: {test_ch:.4f} (클수록 높은 분산 대비 응집을 의미)\n")

    def visualize(self, save_path=None):
        print("5. Step 3: Visual Analytics 생성 중...")
        # 1x3 Subplot 구조: [K-Means 산점도] [실루엣 스코어 흐름] [CH 스코어 흐름]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # --- (1) K-Means 산점도 ---
        ax1 = axes[0]
        ax1.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1], c=self.train_labels, cmap='viridis', marker='o', s=50, edgecolor='k', alpha=0.3, label='Train')
        ax1.scatter(self.X_test_pca[:, 0], self.X_test_pca[:, 1], c=self.test_labels, cmap='viridis', marker='D', s=100, edgecolor='red', linewidths=1.5, alpha=1.0, label='Test (Pred)')
        centroids = self.kmeans.cluster_centers_
        ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=300, linewidths=2, edgecolor='black', label='Centroids')
        ax1.set_title("K-Means Clustering Result (K=3)", fontsize=14)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- (2) Silhouette Score 시각화 ---
        ax2 = axes[1]
        ax2.plot(self.k_range, self.sil_scores, marker='o', color='b', linestyle='solid', linewidth=2, markersize=8)
        ax2.axvline(x=3, color='r', linestyle='--', label='Selected K=3')
        ax2.set_title("Silhouette Score vs Number of Clusters", fontsize=14)
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        # --- (3) Calinski-Harabasz Score 시각화 ---
        ax3 = axes[2]
        ax3.plot(self.k_range, self.ch_scores, marker='s', color='g', linestyle='solid', linewidth=2, markersize=8)
        ax3.axvline(x=3, color='r', linestyle='--', label='Selected K=3')
        ax3.set_title("Calinski-Harabasz Score vs Number of Clusters", fontsize=14)
        ax3.set_xlabel("Number of Clusters (K)")
        ax3.set_ylabel("Calinski-Harabasz Score")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"   [완료] 시각화 이미지 저장 완료: {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    pipeline = PCAKMeansPipeline()
    pipeline.run_pipeline()
    
    # 현재 스크립트가 위치한 폴더(실행 폴더)에 이미지 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "pca_kmeans_pipeline_result.png")
    
    pipeline.visualize(save_path=img_path)
