import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings if any (for clean output execution)
warnings.filterwarnings('ignore', category=UserWarning)

class KMeansAnalysisTask:
    """
    K-Means Clustering Analysis Task following Scikit-learn standard guidelines.
    """
    def __init__(self):
        self.X = None
        self.y_true = None
        self.kmeans = None
        self.labels_ = None
        
    def generate_data(self):
        """
        1. Data Generation
        - n_samples: 300, n_features: 2
        - centers: 5, cluster_std: 0.60
        - random_state: 42
        """
        print("Generating synthetic data...")
        self.X, self.y_true = make_blobs(
            n_samples=300,
            n_features=2,
            centers=5,
            cluster_std=0.60,
            random_state=42
        )
        print(f"Data generation complete. Shape: {self.X.shape}\n")
        
    def run_kmeans(self):
        """
        2. K-Means Implementation
        - n_clusters: 5
        - init: 'k-means++'
        - n_init: 'auto'
        - max_iter: 300
        """
        print("Initializing and training KMeans model...")
        self.kmeans = KMeans(
            n_clusters=5,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            random_state=42  # Adding random_state for reproducible results
        )
        self.labels_ = self.kmeans.fit_predict(self.X)
        print("Model training complete.\n")
        
    def evaluate_and_visualize(self):
        """
        3. Evaluation & Visualization
        - Print Inertia and Silhouette Score
        - Scatter plot with assigned cluster colors and marked centroids
        """
        # --- Evaluation ---
        inertia = self.kmeans.inertia_
        silhouette_avg = silhouette_score(self.X, self.labels_)
        
        print("--- Model Evaluation ---")
        print(f"💡 Inertia (Sum of squared distances): {inertia:.4f}")
        print(f"💡 Silhouette Score: {silhouette_avg:.4f}\n")
        
        # --- Visualization ---
        print("Visualizing the clusters...")
        plt.figure(figsize=(10, 6))
        
        # Plot data points with cluster colors
        plt.scatter(
            self.X[:, 0], self.X[:, 1], 
            c=self.labels_, 
            cmap='viridis', 
            s=50, 
            edgecolor='k', 
            alpha=0.8,
            label='Data Points'
        )
        
        # Plot centroids with a distinct marker
        centroids = self.kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0], centroids[:, 1], 
            c='red', 
            marker='X', 
            s=250, 
            linewidths=3, 
            edgecolor='black',
            label='Centroids'
        )
        
        plt.title("K-Means Clustering Result (k=5)", fontsize=16)
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def print_expert_opinion(self):
        """
        4. Expert Opinion on k-means++ vs random initialization
        """
        opinion = """
================================================================================
[전문가 리뷰: k-means++ 초기화 방식이 무작위(Random) 선택보다 우수한 이유]

전통적인 방식(init='random')은 초기 군집의 중심을 완전히 무작위로 선택합니다. 
이 경우, 초기 중심점들이 우연히 한 곳에 몰리거나 이상치(Outlier)에 가깝게 선택되면 
알고리즘이 '지역 최적해(Local Minimum)'에 빠질 위험이 높으며, 최종 수렴까지 탐색 탭(Iterations)
이 크게 증가하는 비효율성이 발생합니다.

반면, Scikit-learn의 권장 방식인 'k-means++'는 다음과 같은 전략을 사용합니다:
1. 첫 번째 중심점은 데이터 중 무작위로 하나를 선택.
2. 다음 중심점들은 이전에 선택된 중심점들과의 '거리의 제곱에 비례하는 확률'을 
   기반으로 멀리 떨어진 포인트를 우선적으로 선택.

이러한 초기화 방식은 중심점들이 데이터 공간 내에서 최대한 분산되도록 보장합니다.
결과적으로 군집화의 최종 성능(Inertia 최소화)이 크게 향상되며, 최적해에 도달하기 위한 
반복 연산 횟수(Iteration)와 시간을 획기적으로 줄여주기 때문에 최신 라이브러리들에서 
기본(Default) 속성으로 널리 사용됩니다.
================================================================================
"""
        print(opinion)


if __name__ == "__main__":
    # Task 실행
    task = KMeansAnalysisTask()
    
    task.generate_data()
    task.run_kmeans()
    task.evaluate_and_visualize()
    task.print_expert_opinion()
