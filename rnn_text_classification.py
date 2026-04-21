import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

# 1. 텍스트 리뷰 자료 지정
docs = [
    '너무 재밌네요',
    '최고예요',
    '참 잘 만든 영화예요',
    '추천하고 싶은 영화입니다.',
    '한 번 더 보고싶네요',
    '글쎄요',
    '별로예요',
    '생각보다 지루하네요',
    '연기가 어색해요',
    '재미없어요'
]

# 2. 긍정 리뷰는 1, 부정 리뷰는 0으로 클래스 지정 (class 예약어 대신 labels 사용)
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# 3. 토큰화 (Tokenizer)
token = Tokenizer()
token.fit_on_texts(docs)
print("단어 인덱스:\n", token.word_index)

# 텍스트를 숫자로 이루어진 시퀀스로 변환
sequences = token.texts_to_sequences(docs)
print("\n시퀀스 변환 결과:\n", sequences)

# 4. 패딩 (pad_sequences)
# 시퀀스의 길이를 동일하게 맞추기 위해 가장 긴 문장의 길이에 맞춰 패딩 처리
max_len = max(len(s) for s in sequences)
padded_x = pad_sequences(sequences, maxlen=max_len)
print("\n패딩 결과:\n", padded_x)

# 고유 단어의 개수 (인덱스 0을 보존하기 위해 +1)
word_size = len(token.word_index) + 1

# 5. 모델 구축 (SimpleRNN 활용)
model = Sequential()
# Embedding 층: (단어 사전 크기, 임베딩 차원, 입력 시퀀스 길이)
model.add(Embedding(input_dim=word_size, output_dim=8, input_length=max_len))
# SimpleRNN 층: (순환 노드 수)
model.add(SimpleRNN(8))
# 출력 층: 이진 분류이므로 sigmoid 활성화 함수를 사용하는 Dense 층
model.add(Dense(1, activation='sigmoid'))

# 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
print("\n--- 모델 학습 시작 ---")
model.fit(padded_x, labels, epochs=50, verbose=1)

# 6. 새로운 데이터 예측
new_text = ["최고 보고싶다"]

# 훈련 데이터에 적용했던 Tokenizer를 이용해 시퀀스 변환 및 패딩 수행
new_seq = token.texts_to_sequences(new_text)
new_padded = pad_sequences(new_seq, maxlen=max_len)

print("\n--- 새로운 데이터 예측 ---")
print(f"문장: {new_text}")
print(f"시퀀스: {new_seq}")
print(f"패딩 결과: {new_padded}")

# 예측 결과 출력 (확률값)
prediction = model.predict(new_padded)
print(f"\n분석 결과: '{new_text[0]}' 문장이 긍정 리뷰일 확률은 {prediction[0][0]*100:.2f}% 입니다.")
