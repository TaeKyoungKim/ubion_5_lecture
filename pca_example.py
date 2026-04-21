import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# 1. 10개의 피처를 가진 임의의 데이터 생성 (샘플 100개)
# 의미 있는 피처 생성과 상관관계를 만들기 위해 make_classification 사용
X, _ = make_classification(n_samples=100, n_features=10, 
                           n_informative=5, n_redundant=2, 
                           random_state=42)

print("=== [ 원본 데이터 정보 ] ===")
print(f"원본 데이터 형태(shape): {X.shape}")
print("첫 번째 샘플의 특징값 10개:\n", np.round(X[0], 3))
print("-" * 50)

# 2. PCA를 이용한 주성분 추출 (차원 축소: 10차원 -> 3차원)
n_components = 3
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

print("\n=== [ PCA 차원 축소 결과 ] ===")
print(f"축소된 데이터 형태(shape): {X_pca.shape}")
print("첫 번째 샘플의 추출된 피처 3개:\n", np.round(X_pca[0], 3))
print("-" * 50)

# 3. 고유값(Eigenvalues) 및 분산 설명 비율(Variance Ratio) 확인
# pca.explained_variance_ : 데이터 공분산 행렬의 고유값 (추출된 주성분의 분산)
# pca.explained_variance_ratio_ : 전체 데이터 대비 각 주성분이 설명하는 정보의 비율
eigenvalues = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_

print("\n=== [ 고유값(Eigenvalues) 및 설명력 정보 ] ===")
for i in range(n_components):
    print(f"■ 주성분(PC) {i+1}:")
    print(f"   - 고유값(Eigenvalue): {eigenvalues[i]:.4f}")
    print(f"   - 데이터 설명력(분산 비율): {variance_ratio[i]*100:.2f}%")

total_variance = sum(variance_ratio) * 100
print(f"\n=> 결론: 추출된 {n_components}개의 피처가 원본 10개 피처가 가진 전체 데이터 정보 중 총 {total_variance:.2f}%를 설명하고 있습니다.")
