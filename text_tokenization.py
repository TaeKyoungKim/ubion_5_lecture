import numpy
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding

#주어진 문장을 '단어'로 토큰화 하기
#케라스의 텍스트 전처리와 관련한 함수중 text_to_word_sequence 함수를 불러 옵니다.
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# 전처리할 텍스트를 정합니다.
text = '해보지 않으면 해낼 수 없다'

# 해당 텍스트를 토큰화 합니다.
result = text_to_word_sequence(text,split=' ')
print("\n원문:\n", text)
print("\n토큰화:\n", result)

#단어 빈도수 세기
#전처리 하려는 세개의 문장을 정합니다.
docs = [ '먼저 텍스트의 각 단어를 나누어 토큰화 합니다.',
        '텍스트의 단어로 토큰화 해야 딥러닝에서 인식됩니다.',
        '토큰화 한 결과는 딥러닝에서 사용 할 수 있습니다.',
        ]

# 토큰화 함수를 이용해 전처리 하는 과정입니다.
token = Tokenizer()          # 토큰화 함수 지정
token.fit_on_texts(docs)     # 토큰화 함수에 문장 적용

#단어의 빈도수를 계산한 결과를 각 옵션에 맞추어 출력합니다.

print("\n단어 카운트:\n", token.word_counts)
#Tokenizer()의 word_counts 함수는 순서를 기억하는 OrderedDict클래스를 사용합니다.

#출력되는 순서는 랜덤입니다.
print("\n문장 카운트: ", token.document_count)
print("\n각 단어가 몇개의 문장에 포함되어 있는가:\n", token.word_docs)
print("\n각 단어에 매겨진 인덱스 값:\n",  token.word_index)
