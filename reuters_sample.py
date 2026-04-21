import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import reuters

def get_random_reuters_article():
    # 데이터셋 로드
    # num_words 매개변수를 지정하지 않으면 모든 단어를 불러옵니다.
    print("로이터 뉴스 데이터셋을 불러오는 중입니다...")
    (x_train, y_train), (x_test, y_test) = reuters.load_data()
    
    # 단어 인덱스 사전 가져오기 (단어 -> 정수)
    word_index = reuters.get_word_index()
    
    # 정수 인덱스를 다시 단어로 변환하기 위한 사전 생성 (정수 -> 단어)
    # 인덱스는 3만큼 이동되어 있습니다 (0: 패딩, 1: 문서 시작, 2: OOV(Out Of Vocabulary))
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # 훈련 데이터에서 임의의 기사 인덱스 선택
    random_idx = np.random.randint(0, len(x_train))
    article_indices = x_train[random_idx]
    
    # 정수 시퀀스를 텍스트로 디코딩
    # 인덱스에서 3을 빼고, 사전에 없는 경우 '?'를 출력합니다.
    decoded_article = ' '.join([reverse_word_index.get(i - 3, '?') for i in article_indices])
    
    # 레이블 이름 가져오기
    label_names = reuters.get_label_names()
    print(f"\n총 레이블의 종류 (총 {len(label_names)}개):")
    print(label_names)
    
    print(f"\n기사 인덱스: {random_idx}")
    print(f"기사 카테고리 (라벨): {y_train[random_idx]} ({label_names[y_train[random_idx]]})")
    print("\n--- 디코딩되지 않은 기사 내용 (정수 시퀀스) ---")
    print(article_indices)
    print("\n--- 디코딩된 기사 내용 (원문) ---")
    print(decoded_article)

if __name__ == "__main__":
    get_random_reuters_article()
