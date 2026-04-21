import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape, UpSampling2D, Conv2D, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def build_encoder():
    # DCGAN의 판별자(Discriminator) 형태를 본떠 입력 이미지를 압축하는 인코더 구성
    encoder = Sequential(name="Encoder")
    encoder.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    encoder.add(LeakyReLU(0.2))
    encoder.add(Dropout(0.3))
    
    encoder.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    encoder.add(LeakyReLU(0.2))
    encoder.add(Dropout(0.3))
    
    encoder.add(Flatten())
    # 이미지를 100개의 주요 특징 벡터(Latent Space)로 압축합니다.
    encoder.add(Dense(100)) 
    
    return encoder

def build_decoder():
    # DCGAN의 생성자(Generator) 형태를 본떠 압축된 정보를 다시 이미지로 복원하는 디코더 구성
    decoder = Sequential(name="Decoder")
    decoder.add(Dense(128*7*7, input_shape=(100,), activation=LeakyReLU(0.2)))
    decoder.add(BatchNormalization())
    decoder.add(Reshape((7, 7, 128)))
    
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, kernel_size=5, padding='same'))
    decoder.add(BatchNormalization())
    decoder.add(Activation(LeakyReLU(0.2)))
    
    decoder.add(UpSampling2D())
    # 원본과 오차를 줄여나가야 하므로 출력 형태를 동일하게 tanh로 지정
    decoder.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
    
    return decoder

def train_autoencoder():
    # 이미지 저장 경로 설정
    save_dir = 'autoencoder_images'
    os.makedirs(save_dir, exist_ok=True)
    print(f"생성된 이미지는 '{save_dir}' 폴더에 저장됩니다.")

    # 1. 인코더와 디코더 구축
    encoder = build_encoder()
    decoder = build_decoder()

    # 2. 오토인코더(Autoencoder) 전체 모델 결합 
    # 입력 이미지 -> 인코더(압축) -> 디코더(복원) -> 출력 이미지
    autoencoder_input = Input(shape=(28, 28, 1))
    encoded_img = encoder(autoencoder_input)
    reconstructed_img = decoder(encoded_img)
    
    autoencoder = Model(autoencoder_input, reconstructed_img)
    
    # 원본 이미지와 복원된 이미지 간의 픽셀단위 오차(MSE)를 최소화하도록 컴파일
    optim = Adam(learning_rate=0.0002)
    autoencoder.compile(loss='mse', optimizer=optim)

    # 3. 데이터 불러오기 및 전처리
    print("MNIST 데이터셋 로드 중...")
    (X_train, _), (X_test, _) = mnist.load_data()
    
    # -1.0 ~ 1.0 범주로 만들기 (tanh 출력을 고려)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=3)

    # 학습 설정
    epochs = 30
    batch_size = 128
    
    # 평가를 위한 고정 테스트 샘플 추출 (복원 과정을 확인하기 위해 10개 추출)
    num_samples = 10
    fixed_test_images = X_test[:num_samples]

    print("본격적인 학습을 시작합니다!")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # 오토인코더는 자기 자신을 입력과 라벨(정답)로 동시에 사용합니다.
        autoencoder.fit(X_train, X_train, 
                        epochs=1, 
                        batch_size=batch_size, 
                        validation_split=0.1)
        
        # 고정 테스트 샘플 복원 실험
        reconstructed = autoencoder.predict(fixed_test_images, verbose=0)
        
        # 윗줄: 원본 이미지 / 아랫줄: 복원(디코딩) 이미지
        fig = plt.figure(figsize=(num_samples * 1.5, 3))
        for i in range(num_samples):
            # 윗줄 (원본)
            plt.subplot(2, num_samples, i + 1)
            plt.title("Original")
            plt.imshow(fixed_test_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            
            # 아랫줄 (복원)
            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.title("Recon")
            plt.imshow(reconstructed[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'image_at_epoch_{epoch+1:04d}.png'))
        plt.close(fig)

    print("학습 완료!")

if __name__ == '__main__':
    train_autoencoder()
