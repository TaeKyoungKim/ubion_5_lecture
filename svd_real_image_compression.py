import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import TruncatedSVD
import os

# 1. 실제 이미지 로드
image_path = r"c:\apps\ubion_5_lecture\data\original_face.png"

# 파일이 존재하는지 로드 전 확인
if not os.path.exists(image_path):
    print(f"Error: 이미지를 찾을 수 없습니다. 경로를 확인해주세요: {image_path}")
    exit()

img = mpimg.imread(image_path)

# PNG 이미지는 투명도(Alpha)를 포함해 (Height, Width, 4) 채널을 가질 수 있습니다.
# 이 경우 SVD 연산을 위해 빛의 3원색인 RGB 3채널만 사용하도록 잘라냅니다.
if len(img.shape) == 3 and img.shape[2] == 4:
    img = img[:, :, :3]

# 2. 압축할 수준(k값) 설정
# 실제 사람 얼굴 등은 디테일이 많기 때문에 k값을 이전 노이즈보다 약간 더 넉넉하게 줍니다.
k_values = [10, 50, 150] 

# 원본 이미지 용량(요소의 개수) 계산
if len(img.shape) == 2:
    original_size = img.shape[0] * img.shape[1]
    channels = 1
else:
    original_size = img.shape[0] * img.shape[1] * img.shape[2]
    channels = img.shape[2]

print(f"[[ 원본 이미지 정보 ]]")
print(f"해상도: {img.shape}")
print(f"데이터 용량(요소 개수): {original_size:,}\n")

plt.figure(figsize=(16, 5))

# 원본 출력
plt.subplot(1, 4, 1)
# 흑백 이미지일 경우를 대비해 처리
if len(img.shape) == 2:
    plt.imshow(img, cmap='gray')
else:
    plt.imshow(img)
plt.title(f"Original Image\nSize: {original_size:,}")
plt.axis('off')

# 3. 채널별로 SVD 압축을 적용하는 함수
def compress_svd(image, k):
    # 흑백 이미지(2차원 배열)인 경우 
    if len(image.shape) == 2:
        svd = TruncatedSVD(n_components=k, random_state=42)
        compressed = svd.fit_transform(image)
        return svd.inverse_transform(compressed)
    
    # 컬러 이미지(3차원 배열)인 경우 R, G, B 각각 분리해서 SVD 수행 후 합침
    elif len(image.shape) == 3:
        reconstructed_channels = []
        for i in range(image.shape[2]):
            svd = TruncatedSVD(n_components=k, random_state=42)
            # 단일 채널 추출 및 압축
            compressed = svd.fit_transform(image[:, :, i])
            # 채널 복원
            reconstructed_ch = svd.inverse_transform(compressed)
            reconstructed_channels.append(reconstructed_ch)
        
        # 3개의 복원된 2차원 배열 채널들을 하나(3차원 배열)로 다시 쌓기
        return np.stack(reconstructed_channels, axis=2)

# 4. 압축 및 재구성하여 시각화 적용
for i, k in enumerate(k_values):
    # 원본 이미지의 크기(Rank)보다 k값이 클 수 없으므로 안전장치 설정
    rank = min(img.shape[0], img.shape[1])
    current_k = min(k, rank)
    
    # SVD 압축
    reconstructed_img = compress_svd(img, current_k)
    
    # SVD 압축 후 필요한 저장 공간(요소의 개수) 계산
    # (U 행렬 : Height * k) + (VT 행렬 : k * Width) => 각 채널별로 필요
    compressed_size = (img.shape[0] * current_k + current_k * img.shape[1]) * channels
    ratio = (compressed_size / original_size) * 100
    
    print(f"[[ SVD 압축: k={current_k} ]]")
    print(f"데이터 용량(요소 개수): {compressed_size:,} ({ratio:.1f}% of original)\n")

    # matplotlib에서 이미지를 출력할 때 픽셀 값이 범위를 벗어나면 에러나 빈틈이 납니다.
    # mpimg 라이브러리는 float 형태일 때 0.0 ~ 1.0 값을 가지므로 이를 벗어난 값을 잘라냅니다(Clip).
    if img.max() <= 1.0:
        reconstructed_img = np.clip(reconstructed_img, 0.0, 1.0)
    else:
        # 0 ~ 255 정수형 픽셀 데이터일 경우
        reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)

    plt.subplot(1, 4, i + 2)
    if len(img.shape) == 2:
        plt.imshow(reconstructed_img, cmap='gray')
    else:
        plt.imshow(reconstructed_img)
    plt.title(f"Compressed: k={current_k}\nSize: {ratio:.1f}%")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("Real Image Compression using sklearn TruncatedSVD", fontsize=16, y=1.05)

# 압축된 결과 비교 화면으로 저장
output_file = 'real_image_svd_result.png'
plt.savefig(output_file, bbox_inches='tight')
print(f"압축 시각화 이미지가 '{output_file}'로 저장되었습니다.")
plt.show()
