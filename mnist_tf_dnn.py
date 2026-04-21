import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# 1. 텐서플로우를 이용해 MNIST 데이터 로드 및 전처리
print("MNIST 데이터를 불러오고 있습니다...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 정규화 (스케일링): 픽셀 값을 0 ~ 255에서 0.0 ~ 1.0으로 변환시켜 학습 성능 향상
X_train = X_train / 255.0
X_test = X_test / 255.0

# 2. 모델 구조 설계 (DNN)
# 사용자가 요청한: 히든 레이어 3개 (64개, 32개, 16개) 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),          # 입력층: 28x28 이차원 이미지를 1차원(784)으로 펴줌
    Dense(64, activation='relu'),           # 첫 번째 은닉층: 뉴런 64개
    Dense(32, activation='relu'),           # 두 번째 은닉층: 뉴런 32개
    Dense(16, activation='relu'),           # 세 번째 은닉층: 뉴런 16개
    Dense(10, activation='softmax')         # 출력층: 10개 (숫자 0~9 확률 예측)
])

# 모델 요약 정보 출력
model.summary()

# 3. 모델 컴파일 및 학습
# 손실함수: 정수형 레이블이므로 sparse_categorical_crossentropy 사용
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nDNN 모델 학습을 시작합니다...")
# 학습 반복 횟수(epochs)는 5로 설정
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
print("학습 완료!\n")

# 4. 테스트 셋으로 평가
print("테스트 셋으로 모델을 평가합니다...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 셋 전체 정확도 (Accuracy): {test_acc * 100:.2f}%\n")

# 5. 테스트 셋 중 10번째 데이터 확인 (인덱스는 0부터 시작하므로 9)
target_idx = 10
target_image = X_test[target_idx]
actual_label = y_test[target_idx]

# 예측 값을 구하려면 입력 차원을 3차원 (샘플개수, 높이, 너비) 으로 맞춰줘야 합니다.
prediction_probs = model.predict(target_image.reshape(1, 28, 28))
# 예측 확률들 중 가장 높은 확률을 가진 클래스가 모델이 예측한 숫자입니다.
predicted_class = np.argmax(prediction_probs)

print("\n" + "=" * 45)
print("         테스트 셋 10번째 데이터 예측 결과          ")
print("=" * 45)
print(f"- 실제 숫자 (정답): {actual_label}")
print(f"- 텐서플로우 DNN 모델의 예측 숫자: {predicted_class}")
print("=" * 45)

# 10번째 이미지를 화면에 시각화 (선택적)
plt.imshow(target_image, cmap='gray')
plt.title(f"TensorFlow DNN => Predicted: [{predicted_class}], Actual: [{actual_label}]", fontsize=14)
plt.axis('off')
plt.show()
