import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# 1. 임의의 가상 이미지 생성
# SVD의 효과를 잘 보여주기 위해 완전 무작위 노이즈보다는 패턴이 있는 이미지를 생성합니다.
# 256x256 크기의 흑백 이미지를 만듭니다.
x = np.linspace(0, 10, 256)
y = np.linspace(0, 10, 256)
X, Y = np.meshgrid(x, y)

# 사인파와 코사인파를 섞어서 임의의 패턴 생성
original_image = np.sin(X) * np.cos(Y) + 0.5 * np.sin(3 * X) + 0.3 * np.cos(5 * Y)

# 이미지의 픽셀 값을 0 ~ 255 사이로 정규화합니다.
original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255

# 2. 시각화를 위한 설정
# 비교할 3가지 압축 수준(사용할 상위 특이값의 개수 k)을 설정합니다.
k_values = [5, 20, 50]

plt.figure(figsize=(16, 4)) # 전체 그래프 크기 설정

# 3. 원본 이미지 출력
plt.subplot(1, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title(f"Original Image\n(Shape: {original_image.shape})")
plt.axis('off')

# 4. scikit-learn의 TruncatedSVD를 이용해 이미지 압축 및 재구성 출력
for i, k in enumerate(k_values):
    # TruncatedSVD 모델 객체 생성 (n_components 속성에 k값 지정)
    svd = TruncatedSVD(n_components=k, random_state=42)
    
    # fit_transform: 원본 데이터를 학습하고 k개의 차원으로 압축
    # 생성되는 차원 크기는 (Original Height, k)가 됩니다.
    compressed_features = svd.fit_transform(original_image)
    
    # inverse_transform: 압축된 데이터를 다시 원래 차원인 256 크기로 복원
    # 역변환된 이미지는 근사된 이미지 행렬(Original Height, Original Width)이 됩니다.
    reconstructed_image = svd.inverse_transform(compressed_features)
    
    # 복원된 픽셀 값이 정상 범위를 벗어나지 않게 클리핑 (옵션)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    # 데이터의 압축된 사이즈 및 비율 계산
    # compressed_features (256*k) 크기와 svd.components_ (k*256) 메모리 크기를 합산
    original_size = 256 * 256
    compressed_size = 256 * k + k * 256
    ratio = (compressed_size / original_size) * 100

    plt.subplot(1, 4, i + 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Compressed: k={k}\nSize: {ratio:.1f}%")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("Image Compression using sklearn TruncatedSVD", fontsize=16, y=1.05)

# 이미지 저장 및 화면 출력
plt.savefig('sklearn_svd_compression_result.png', bbox_inches='tight')
plt.show()
