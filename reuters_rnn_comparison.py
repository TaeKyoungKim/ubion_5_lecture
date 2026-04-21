import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

def compare_rnn_models():
    # 1. 데이터 설정 및 전처리
    max_features = 10000  # 가장 빈도수가 높은 10,000개의 단어만 사용
    maxlen = 200          # 각 기사의 최대 길이를 200 단어로 제한 (패딩/절단)
    batch_size = 128
    epochs = 5
    
    print("로이터 데이터셋을 불러오는 중...")
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
    print(f"훈련 데이터 수: {len(x_train)}")
    print(f"테스트 데이터 수: {len(x_test)}")
    
    print(f"시퀀스 패딩 진행 (길이: {maxlen})...")
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    # 원-핫 인코딩 처리를 위한 클래스 수 확인
    num_classes = int(np.max(y_train) + 1)
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # 모델 생성 함수
    def create_model(rnn_type='lstm'):
        model = Sequential()
        model.add(tf.keras.layers.Input(shape=(maxlen,)))
        model.add(Embedding(max_features, 128))
        if rnn_type == 'lstm':
            model.add(LSTM(64))
        elif rnn_type == 'gru':
            model.add(GRU(64))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return model

    print("\n==================================")
    print("      LSTM 모델 학습 시작         ")
    print("==================================")
    lstm_model = create_model('lstm')
    lstm_model.summary()
    
    start_time_lstm = time.time()
    lstm_model.fit(x_train, y_train, 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=0.2, 
                   verbose=1)
    end_time_lstm = time.time()
    lstm_duration = end_time_lstm - start_time_lstm
    
    print("LSTM 테스트 데이터 평가 중...")
    lstm_score = lstm_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    
    print("\n==================================")
    print("      GRU 모델 학습 시작          ")
    print("==================================")
    gru_model = create_model('gru')
    gru_model.summary()
    
    start_time_gru = time.time()
    gru_model.fit(x_train, y_train, 
                  batch_size=batch_size, 
                  epochs=epochs, 
                  validation_split=0.2, 
                  verbose=1)
    end_time_gru = time.time()
    gru_duration = end_time_gru - start_time_gru
    
    print("GRU 테스트 데이터 평가 중...")
    gru_score = gru_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    # 최종 결과 비교 출력
    print("\n")
    print("=" * 45)
    print("            [ 평가 및 비교 결과 ]            ")
    print("=" * 45)
    print(f" 항목 \t\t LSTM \t\t GRU")
    print("-" * 45)
    print(f" 학습 시간(초)\t {lstm_duration:.2f} sec \t {gru_duration:.2f} sec")
    print(f" 테스트 정확도\t {lstm_score[1]:.4f} \t {gru_score[1]:.4f}")
    print(f" 테스트 손실값\t {lstm_score[0]:.4f} \t {gru_score[0]:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    compare_rnn_models()
