import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib

# 한글 폰트 설정 (Windows 기준, 깨짐 방지용)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 시각화를 위한 임의의 데이터 생성 (반달 모양 데이터에 노이즈를 섞음)
X, y = make_moons(n_samples=100, noise=0.3, random_state=42)

# 결정 경계를 그려주는 함수
def plot_decision_boundary(clf, X, y, ax, title):
    # 그래프를 그릴 영역 설정
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 모델 예측을 통해 배경의 색상 영역 결정
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계선 및 색상 채우기
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # 실제 데이터 포인트 표시
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title, fontsize=12)
    ax.set_xticks(())
    ax.set_yticks(())

# 3x3 형태의 그래프 영역 생성
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# -------------------------------------------------------------
# C값과 gamma값의 조합에 따른 영향
# 행(가로): C (0.1, 1, 100)
# 열(세로): gamma (0.1, 1, 10)
# -------------------------------------------------------------
C_values = [0.1, 1, 100]
gamma_values = [0.1, 1, 10]

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        clf = SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X, y)
        
        # 상태에 대한 대략적인 설명
        if C == 0.1 and gamma == 0.1:
            desc = "Underfitting (과소적합)"
        elif C == 1 and gamma == 1:
            desc = "Good Fit (적절합)"
        elif C == 100 and gamma == 10:
            desc = "Extreme Overfitting (극심한 과대적합)"
        elif C == 100 or gamma == 10:
            desc = "Overfitting (과대적합)"
        else:
            desc = ""
            
        title = f'C={C}, gamma={gamma}\n{desc}' if desc else f'C={C}, gamma={gamma}'
        plot_decision_boundary(clf, X, y, axes[i, j], title)

plt.tight_layout()
plt.show()
