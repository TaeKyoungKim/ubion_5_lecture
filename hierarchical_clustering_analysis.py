import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import warnings

# 경고 무시 (일부 scikit-learn/scipy 버전의 Deprecation Warning 등 방지)
warnings.filterwarnings('ignore')

class ClusteringAnalysisManager:
    """
    계층형 클러스터링(Hierarchical Clustering) 분석을 관리하는 클래스입니다.
    데이터 생성, 모델 적용, 평가 및 시각화 파이프라인을 캡슐화하여 재사용성을 확보했습니다.
    """
    def __init__(self, n_samples=250, centers=4, random_state=42):
        self.n_samples = n_samples
        self.centers = centers
        self.random_state = random_state
        self.X = None
        self.y_true = None
        
        # 분석을 진행할 4가지 주요 연결 기법(Linkage Methods)
        self.linkage_methods = ['ward', 'complete', 'average', 'single']
        self.linkage_matrices = {}
        self.silhouette_scores = {}
        
    def generate_data(self):
        """
        scikit-learn의 make_blobs를 사용하여 3~5개의 중심점을 가진 150개 이상의 데이터셋을 만듭니다.
        """
        self.X, self.y_true = make_blobs(
            n_samples=self.n_samples, 
            centers=self.centers, 
            cluster_std=1.2, # 클러스터 간의 겹침을 약간 유도하여 분석 모델 변별력을 높임
            random_state=self.random_state
        )
        print(f"[*] 합성 데이터 생성 완료: 총 샘플 수 {self.n_samples}개, 중심점(Centers) {self.centers}개")

    def perform_clustering(self):
        """
        4가지 연결 기법을 적용하여 계층형 클러스터링을 수행하고, 실루엣 스코어로 평가합니다.
        """
        print("\n[*] 계층형 클러스터링 모형 적합 및 실루엣 계수 평가 중...")
        for method in self.linkage_methods:
            # 거리 행렬 및 군집 간 연결 거리 계산 (scipy.cluster.hierarchy.linkage 활용)
            Z = linkage(self.X, method=method)
            self.linkage_matrices[method] = Z
            
            # 생성한 계층 트리를 기반으로 목표 군집 수(self.centers)만큼 클러스터 커팅
            labels = fcluster(Z, t=self.centers, criterion='maxclust')
            
            # 클러스터가 2개 이상 구축되었을 때만 Silhouette Score 산출 가능
            if len(np.unique(labels)) > 1:
                score = silhouette_score(self.X, labels)
            else:
                score = -1.0
            
            self.silhouette_scores[method] = score
            print(f"    - {method.capitalize():<10} Linkage 실루엣 스코어: {score:.4f}")

    def visualize_dendrograms(self):
        """
        각 기법에 따른 덴드로그램(Dendrogram)을 2x2 서브플롯으로 시각화하여 트리 구조의 차이를 비교합니다.
        """
        print("\n[*] 2x2 덴드로그램 서브플롯 생성 중...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, method in enumerate(self.linkage_methods):
            ax = axes[i]
            # 시각적 가독성을 높이기 위해 truncate_mode='level'을 사용하여 하위 노드를 축약
            dendrogram(
                self.linkage_matrices[method], 
                ax=ax, 
                truncate_mode='level', 
                p=5, 
                show_contracted=True
            )
            ax.set_title(f'Dendrogram: {method.capitalize()} Linkage', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sample Index / (Cluster Size)', fontsize=10)
            ax.set_ylabel('Distance (Cophenetic)', fontsize=10)
            
        plt.tight_layout()
        plt.show() # 첫 번째 화면: 덴드로그램 서브플롯

    def visualize_best_clustering(self):
        """
        가장 뛰어난 실루엣 스코어를 달성한 기법을 선정하고, 군집 형상 결과를 산점도로 시각화합니다.
        """
        # 최대 실루엣 계수를 도출한 기법 탐색
        best_method = max(self.silhouette_scores, key=self.silhouette_scores.get)
        best_score = self.silhouette_scores[best_method]
        
        print(f"\n[*] 최종 최적 Linkage 기법 선정: {best_method.capitalize()} Linkage")
        print(f"[*] 최고 실루엣 수준: {best_score:.4f}")
        
        # 최적의 연결 구조에서 클러스터 레이블 추출
        Z = self.linkage_matrices[best_method]
        best_labels = fcluster(Z, t=self.centers, criterion='maxclust')
        
        # 산점도 시각화 처리
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(
            self.X[:, 0], self.X[:, 1], 
            c=best_labels, 
            cmap='viridis', 
            edgecolor='k', 
            s=60, 
            alpha=0.8
        )
        
        plt.title(f'Best Clustering Result [{best_method.capitalize()} Linkage]\nSilhouette Score: {best_score:.4f}', 
                  fontsize=15, fontweight='bold')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.colorbar(scatter, label='Cluster Label')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show() # 두 번째 화면: 최적 결과 산점도

    def run_pipeline(self):
        """
        데이터 엔지니어링 및 시각화 전 과정을 일괄 실행합니다.
        """
        self.generate_data()
        self.perform_clustering()
        self.visualize_dendrograms()
        self.visualize_best_clustering()
        print("\n[*] 파이프라인 분석이 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    # 요구사항에 따라 150개 이상의 데이터(250), 3~5개 중심점(4)을 배치
    manager = ClusteringAnalysisManager(n_samples=250, centers=4, random_state=42)
    manager.run_pipeline()
