import numpy as np
import matplotlib.pyplot as plt

# Keras나 sklearn을 통해 MNIST 데이터를 불러옵니다.
try:
    from tensorflow.keras.datasets import mnist
    print("tensorflow.keras.datasets 모듈로 MNIST 데이터를 불러옵니다...")
    (X_train, y_train), (_, _) = mnist.load_data()
except ImportError:
    print("tensorflow가 설치되어 있지 않습니다. sklearn으로 데이터를 불러옵니다...")
    from sklearn.datasets import fetch_openml
    mnist_data = fetch_openml('mnist_784', version=1, parser='auto')
    X_train = mnist_data.data.values.reshape(-1, 28, 28)
    y_train = mnist_data.target.astype(int).values

# 레이블이 '5'인 첫 번째 이미지의 인덱스를 찾습니다.
idx = np.where(y_train == 5)[0][0]
img = X_train[idx]

# 그림 크기 설정 (픽셀 값을 표시하기 위해 넉넉한 크기로 지정)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)

# 이미지 시각화 (회색조)
ax.imshow(img, cmap='gray')
width, height = img.shape

# 텍스트 색상을 결정하기 위한 임계값 (밝은 픽셀엔 어두운 글씨, 어두운 픽셀엔 밝은 글씨)
thresh = img.max() / 2.5

# 각 픽셀에 해당하는 값을 이미지 위에 텍스트로 표시
for x in range(width):
    for y in range(height):
        val = img[x, y]
        # 값이 0인 배경 픽셀은 생략하여 가독성을 높일 수도 있지만, 
        # 원본 그대로 모든 픽셀을 보고 싶다면 아래 코드를 그대로 사용합니다.
        # if val == 0: continue # 배경 0을 숨기고 싶다면 이 주석을 해제하세요.
        ax.annotate(str(int(val)), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if val < thresh else 'black',
                    fontsize=8)

plt.title('MNIST Data Visualization: Target Digit 5 with Pixel Values', fontsize=18)
plt.axis('off') # 축 영역 숨기기
plt.tight_layout()
plt.show()
