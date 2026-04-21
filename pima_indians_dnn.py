import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def main():
    print("=== Pima Indians Diabetes Dataset DNN Classifier ===")
    
    # ---------------------------------------------------------
    # 1. 데이터 로드 및 전처리 (Data Pipeline)
    # ---------------------------------------------------------
    print("\n[1] 데이터 로드 및 전처리를 시작합니다...")
    # URL에서 직접 CSV 데이터 로드 (헤더가 없는 데이터이므로 커스텀 컬럼명 지정)
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)

    # 0으로 채워진 결측치를 중앙값으로 처리
    # Glucose, BloodPressure, SkinThickness, Insulin, BMI는 0이 될 수 없는 생체 지표이므로 0을 NaN으로 치환
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_features] = df[zero_features].replace(0, np.nan)

    # 각 컬럼별 중앙값(median) 계산 후 NaN 대체
    for feature in zero_features:
        median_val = df[feature].median()
        df[feature] = df[feature].fillna(median_val)

    # 피처(X)와 타겟(y) 데이터 분리
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 학습/테스트 데이터 8:2 분할 (stratify=y를 통해 클래스 불균형 유지)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # StandardScaler를 통한 피처 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # 2. DNN 모델 아키텍처 (Model Architecture)
    # ---------------------------------------------------------
    print("\n[2] 모델 아키텍처를 구성합니다...")
    model = Sequential()

    # 입력층 & 첫 번째 은닉층
    model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 두 번째 은닉층
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 세 번째 은닉층
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 출력층 (이진 분류: Sigmoid)
    model.add(Dense(1, activation='sigmoid'))

    # ---------------------------------------------------------
    # 3. 학습 설정 (Compile & Fit)
    # ---------------------------------------------------------
    print("\n[3] 모델 컴파일 및 학습을 시작합니다...")
    # 최신 Keras 문법에 맞춰 lr 대신 learning_rate 사용
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # EarlyStopping 콜백 정의
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )

    # 모델 학습 (Validation Split 0.2 적용)
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,          # 충분히 큰 에포크 (EarlyStopping 작동 예정)
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # ---------------------------------------------------------
    # 4. 평가 및 시각화 (Evaluation & Visualization)
    # ---------------------------------------------------------
    print("\n[4] 모델 평가 및 결과를 분석합니다...")
    # Test 데이터 기반 최종 평가
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"-> [Test Evaluation] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # 향후 모델 출력 분포를 위한 이진 분류 결과 예측
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion Matrix 및 Classification Report
    print("\n" + "="*50)
    print("Confusion Matrix:")
    print("-" * 50)
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*50)
    print("Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print("="*50)

    # 학습 과정 시각화 (Matplotlib)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy 추이 그래프
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    ax1.set_title('Model Accuracy History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Loss 추이 그래프
    ax2.plot(history.history['loss'], label='Train Loss', color='red')
    ax2.plot(history.history['val_loss'], label='Val Loss', color='green')
    ax2.set_title('Model Loss History')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
