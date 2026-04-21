import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)

class PCAKMeansTask:
    """
    Dimensionality Reduction (PCA) followed by Clustering (K-Means)
    Implementation on the Iris Dataset adhering to Scikit-learn Best Practices.
    """
    def __init__(self):
        self.X_raw = None
        self.y_true = None
        self.feature_names = None
        self.X_scaled = None
        self.X_pca = None
        self.pca = None
        self.kmeans = None
        self.labels_ = None
        
    def data_preparation(self):
        """
        1. Data Preparation
        - Load Iris Data
        - Standardize features using StandardScaler
        """
        print("1. Data Preparation...")
        iris = load_iris()
        self.X_raw = iris.data
        self.y_true = iris.target
        self.feature_names = iris.feature_names
        
        # Scaling is EXTREMELY important before PCA
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X_raw)
        print(f"✅ Data loaded and Standardized. Shape: {self.X_scaled.shape}\n")
        
    def apply_pca(self):
        """
        2. Step 1: PCA (Dimensionality Reduction)
        - n_components=2
        - Output explained_variance_ratio_
        """
        print("2. Step 1: PCA (Dimensionality Reduction)...")
        # Extract 2 Principal Components
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        variance_ratio = self.pca.explained_variance_ratio_
        print("✅ Explained Variance Ratio by components:")
        print(f"   - Principal Component 1 (PC1): {variance_ratio[0]*100:.2f}%")
        print(f"   - Principal Component 2 (PC2): {variance_ratio[1]*100:.2f}%")
        print(f"   👉 Total Explained Variance: {sum(variance_ratio)*100:.2f}%\n")
        
    def apply_kmeans(self):
        """
        3. Step 2: K-Means (Clustering)
        - Use 2D PCA data
        - n_clusters=3
        - init='k-means++', n_init='auto'
        """
        print("3. Step 2: K-Means (Clustering)...")
        self.kmeans = KMeans(
            n_clusters=3,
            init='k-means++',
            n_init='auto',
            random_state=42 # added for reproducibility
        )
        self.labels_ = self.kmeans.fit_predict(self.X_pca)
        print("✅ K-Means modeling complete.\n")
        
    def visualize_and_analyze(self):
        """
        4. Visualization
        - 2D Scatter plot with PCA components
        - Color distinct clusters
        - Mark centroids
        """
        print("4. Generating Visualization...")
        plt.figure(figsize=(10, 6))
        
        # Plot the data points colored by KMeans Cluster label
        plt.scatter(
            self.X_pca[:, 0], self.X_pca[:, 1], 
            c=self.labels_, 
            cmap='viridis', 
            s=70, 
            edgecolor='k', 
            alpha=0.8,
            label='Data Points'
        )
        
        # Plot the Centroids
        centroids = self.kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0], centroids[:, 1], 
            c='red', 
            marker='X', 
            s=300, 
            linewidths=3, 
            edgecolor='black',
            label='Centroids'
        )
        
        plt.title("PCA followed by K-Means Clustering Result (Iris Dataset)", fontsize=16)
        plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
        plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def print_expert_opinion(self):
        """
        4-2. Expert Opinion on PCA and Clustering Integration
        """
        opinion = """
================================================================================
[전문가 리뷰: PCA 차원 축소가 클러스터링에 미치는 영향 및 시너지 효과]

1. 가시성 (Visibility) 및 해석력 향상
원본 Iris 데이터는 4차원(Sepal Length, Width, Petal Length, Width)이므로
인간의 오감으로 한 번에 데이터의 분포와 군집을 시각적으로 파악하는 것은 불가능에 
가깝습니다. PCA를 통해 데이터의 가장 핵심적인 정보(분산)를 보존하며 2차원으로 
압축하면 산점도를 통해 군집의 형성 경계와 크기를 직관적으로 모니터링할 수 
있습니다 (코드 실행 결과 참조).

2. 노이즈 제거와 차원의 저주(Curse of Dimensionality) 완화
K-Means와 같이 '거리 계산'에 기반한 알고리즘은 고차원 공간일수록 각 포인트 간의
거리가 무의미해질 수 있습니다. 이번 실습의 결과처럼 2개의 주성분(PC1, PC2)만으로도 
전체 데이터 분산의 약 95% 이상을 훌륭하게 설명할 수 있습니다.
즉, 남은 5%의 분산(노이즈, 혹은 덜 중요한 정보)을 버리는 최적화 과정을 통해:
 - 거리 계산에 필요 없는 노이즈를 제거하여 군집의 응집도(Cohesion)가 향상됩니다.
 - 연산 속도(Computational Complexity)가 단축됩니다.
 - 피처들이 강한 상관관계를 가질 때 발생하는 다중공선성(Multicollinearity) 문제를 해결합니다.

결론적으로, StandardScaler 스케일링 -> PCA 차원 축소 -> K-Means 클러스터링의
파이프라인 구축은 안정적이고 명료한 비지도 학습 분석을 진행하는 표준(Best Practice)입니다.
================================================================================
"""
        print(opinion)

if __name__ == "__main__":
    task = PCAKMeansTask()
    task.data_preparation()
    task.apply_pca()
    task.apply_kmeans()
    task.visualize_and_analyze()
    task.print_expert_opinion()
