import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, UpSampling2D, Conv2D, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def build_generator():
    # 첨부해주신 이미지의 코드(강의 교안/자료)를 정확히 반영합니다.
    generator = Sequential()
    generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))
    generator.add(BatchNormalization())
    generator.add(Reshape((7, 7, 128)))
    
    generator.add(UpSampling2D())
    generator.add(Conv2D(64, kernel_size=5, padding='same'))
    generator.add(BatchNormalization())
    generator.add(Activation(LeakyReLU(0.2)))
    
    generator.add(UpSampling2D())
    generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
    
    return generator

def build_discriminator():
    # 업샘플링 기반 DCGAN에 맞춘 판별자
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))  # 진짜(1)/가짜(0) 판별이므로 Sigmoid
    
    return discriminator

def train_gan():
    # 이미지 저장 경로 설정
    save_dir = 'dcgan_images'
    os.makedirs(save_dir, exist_ok=True)
    print(f"생성된 이미지는 '{save_dir}' 폴더에 저장됩니다.")

    # 1. 최적화 도구(Optimizer) 설정
    optim = Adam(learning_rate=0.0002, beta_1=0.5)

    # 2. 판별자 모델 컴파일 
    # (이 컴파일 단계는 `discriminator.train_on_batch`를 호출할 때 반영됩니다)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    # 3. 생성자 모델 구축
    generator = build_generator()

    # 4. 결합 모델(GAN) 구축 (생성자 학습용)
    # 질문하셨던 의문점이 여기서 해소됩니다! 'gan' 모델 안에서만 판별자의 가중치를 동결시킵니다.
    discriminator.trainable = False
    
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=optim)

    # 5. 데이터 불러오기 및 전처리
    print("MNIST 데이터셋 로드 중...")
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # -1.0 ~ 1.0 범주로 만들기 (tanh에 맞춤)
    X_train = np.expand_dims(X_train, axis=3)               # (60000, 28, 28, 1) 로 채널 차원 추가

    # 학습 설정
    epochs = 50
    batch_size = 128
    batch_count = X_train.shape[0] // batch_size
    
    # 에포크마다 동일하게 변화해가는 모습을 관찰하기 위한 고정된 시드(노이즈)
    fixed_noise = np.random.normal(0, 1, (16, 100))

    print("본격적인 학습을 시작합니다!")
    for epoch in range(epochs):
        for i in range(batch_count):
            # === [판별자 학습 과정] ===
            # 여기서 판별자는 업데이트를 '진행' 합니다. (discriminator.trainable = False의 영향을 받지 않는 독립된 컴파일 환경)
            
            # 1. 실제 숫자 이미지 준비 (Labels = 1)
            real_images = X_train[i * batch_size : (i + 1) * batch_size]
            real_labels = np.ones((batch_size, 1))

            # 2. 생성자를 이용해 가짜 이미지 만들기 (Labels = 0)
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            fake_images = generator.predict(noise, verbose=0)
            fake_labels = np.zeros((batch_size, 1))

            # 3. 실제/가짜를 각각 학습
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # 손실/정확도 평균

            # === [생성자 학습 과정] ===
            # 여기서 결합 모델(gan)을 이용하며, 판별자의 가중치는 동결(True)된 상태로 생성자 앞단의 가중치만 업데이트 됩니다.
            noise_for_gan = np.random.normal(0, 1, size=(batch_size, 100))
            
            # 목적: 판별자가 가짜 이미지를 진짜(1)라고 착각하게끔 속인다. (Labels = 1)
            g_loss = gan.train_on_batch(noise_for_gan, np.ones((batch_size, 1)))

        print(f"Epoch {epoch + 1}/{epochs} - [D Loss: {d_loss[0]:.4f}, D Acc: {100 * d_loss[1]:.2f}%] [G Loss: {g_loss:.4f}]")
        
        # 1 에포크를 순회할 때마다 고정 노이즈를 넣어 이미지 생성 후 저장
        generated_images = generator.predict(fixed_noise, verbose=0)
        fig = plt.figure(figsize=(4, 4))
        for k in range(generated_images.shape[0]):
            plt.subplot(4, 4, k+1)
            plt.imshow(generated_images[k, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f'image_at_epoch_{epoch+1:04d}.png'))
        plt.close(fig)

    print("학습 완료!")

if __name__ == '__main__':
    train_gan()
