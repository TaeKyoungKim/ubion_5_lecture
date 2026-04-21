import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.datasets import fashion_mnist

# Fashion MNIST의 실제 10개 클래스 이름 정의
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 1. 데이터 로드 및 정규화
print("Fashion MNIST 데이터를 불러오는 중입니다...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# CNN에 입력하기 위해 차원 확장 (28x28 -> 28x28x1) 및 정규화 (0~1)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# 2. CNN 모델 설계
model = Sequential([
    # 첫 번째 Conv2D & Max Pooling
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # 두 번째 Conv2D & Max Pooling
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # 세 번째 Conv2D & Max Pooling
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # DNN (Fully Connected Layer) 구조
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

print("\n[ CNN 아키텍처 요약 ]")
model.summary()

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nFashion MNIST CNN 모델의 학습을 시작합니다...")
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
print("학습이 성공적으로 완료되었습니다.\n")

# 4. 테스트 데이터를 이용한 평가
print("테스트 셋을 이용해 모델을 평가합니다...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도 (Test Accuracy): {test_acc * 100:.2f}%\n")

# 5. 테스트 셋의 10번째 데이터 예측 (인덱스 9)
target_idx = 9
target_image = X_test[target_idx]
actual_label = y_test[target_idx]
actual_name = class_names[actual_label]

# 모델을 이용해 확률을 예측 (입력 형태는 배치차원을 포함한(1, 28, 28, 1)이어야 함)
prediction_probs = model.predict(target_image.reshape(1, 28, 28, 1))
predicted_class = np.argmax(prediction_probs)
predicted_name = class_names[predicted_class]

print("=" * 50)
print(f"      테스트 셋 10번째 데이터(인덱스 {target_idx}) 예측 결과      ")
print("=" * 50)
print(f"- 실제 정답 라벨: {actual_label} => [{actual_name}]")
print(f"- 모델 예측 라벨: {predicted_class} => [{predicted_name}]")
print("=" * 50)

# 사용자의 시각적 확인을 위해 이미지 출력
plt.imshow(target_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_name} / Actual: {actual_name}", fontsize=14)
plt.axis('off')
plt.show()
