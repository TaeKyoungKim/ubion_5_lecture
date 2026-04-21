import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. MNIST 데이터 로드
try:
    from tensorflow.keras.datasets import mnist
    print("tensorflow.keras.datasets 모듈로 MNIST 데이터를 불러옵니다...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 랜덤 포레스트 학습을 위해 2차원(28x28) 이미지를 1차원(784)으로 변환
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
except ImportError:
    print("tensorflow가 설치되어 있지 않습니다. sklearn으로 데이터를 불러옵니다...")
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    # sklearn의 MNIST는 1차원(784)으로 저장되어 있음
    mnist_data = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist_data.data.values
    y = mnist_data.target.astype(int).values
    
    # 60,000개를 학습 데이터로, 10,000개를 테스트 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# 2. 랜덤 포레스트 모델 학습
print("\n랜덤 포레스트 모델로 학습을 시작합니다... (이 과정은 환경에 따라 몇 초 ~ 1분 정도 소요될 수 있습니다)")
start_time = time.time()
# n_jobs=-1 설정으로 가능한 모든 CPU 코어를 사용하여 학습 속도 향상
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f"학습 완료! (소요 시간: {elapsed_time:.2f}초)")

# 3. 테스트셋으로 평가
print("\n테스트 셋으로 모델 성능을 평가합니다...")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 셋 전체 정확도 (Accuracy): {accuracy * 100:.2f}%")

# 4. 테스트 셋 중 10번째 데이터 예측 (인덱스 번호는 9)
target_idx = 9
# predict 메서드는 2차원 배열을 기대하므로 reshape(1, -1) 처리
prediction = rf_model.predict(X_test[target_idx].reshape(1, -1))[0]
actual_label = y_test[target_idx]

print("\n" + "="*45)
print("             테스트 셋의 10번째 데이터 예측             ")
print("="*45)
print(f"- 실제 숫자 (정답): {actual_label}")
print(f"- 랜덤포레스트 예측: {prediction}")
print("="*45)

# 해당 이미지를 시각화하여 확인 (옵션)
img_to_show = X_test[target_idx].reshape(28, 28)
plt.imshow(img_to_show, cmap='gray')
plt.title(f"RandomForest: Predicted [{prediction}] / Actual [{actual_label}]", fontsize=14)
plt.axis('off')
plt.show()
