import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import matplotlib

# 한글 폰트 설정 (Windows 기준, 깨짐 방지용)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------
# 1. 예제 데이터 생성 (사인 곡선에 노이즈를 추가한 비선형 데이터)
# -------------------------------------------------------------
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)  # 0~5 범위의 무작위 값 100개 정렬
y = np.sin(X).ravel()                            # 사인 곡선 생성
y += np.random.normal(0, 0.2, y.shape)           # 약간의 노이즈 추가

# -------------------------------------------------------------
# 2. 다양한 커널을 사용하는 SVM 회귀(SVR) 모델 설정 및 학습
# - RBF 커널: 비선형 관계를 가장 잘 잡는 기본 커널
# - 선형(Linear) 커널: 단순한 직선형 회귀
# - 다항(Polynomial) 커널: 다항식 기반의 곡선 피팅
# * epsilon 파라미터는 에러를 무시하는 오차 허용 경계의 너비를 뜻합니다.
# -------------------------------------------------------------
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel='linear', C=100)
svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1, coef0=1)

# 학습 및 예측 수행
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# -------------------------------------------------------------
# 3. 회귀 결과 시각화
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))

# 원본 데이터 산점도
plt.scatter(X, y, color='darkorange', label='실제 데이터 (Data)', zorder=10)

# 세 종류의 SVR 모델의 예측결과 선 그래프
plt.plot(X, y_rbf, color='navy', lw=2, label='RBF 커널 회귀')
plt.plot(X, y_lin, color='c', lw=2, label='선형(Linear) 커널 회귀')
plt.plot(X, y_poly, color='cornflowerblue', lw=2, label='다항(Polynomial) 커널 회귀')

plt.xlabel('X (입력값)')
plt.ylabel('y (목표값)')
plt.title('서포트 벡터 머신 회귀 (Support Vector Regression, SVR)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
