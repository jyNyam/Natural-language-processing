# 필요한 라이브러리 설치
# pip install konlpy pandas numpy sklearn tensorflow


# 1. 데이터셋을 만든 후 방언-표준어 쌍 데이터 구성

import pandas as pd

data = {
    '방언': ['밥 묵었나?', '어디 가노?', '뭐하노?'],
    '표준어': ['밥 먹었니?', '어디 가니?', '뭐 하니?']
}
df = pd.DataFrame(data)
df.to_csv('dialect_to_standard.csv', index=False)

# 2. 데이터 전처리, 형태소 분석
from konlpy.tag import Okt

okt = Okt()
sentence = "밥 묵었나?"
print(okt.morphs(sentence))  # ['밥', '묵', '었', '나', '?']


# 3. 토큰화 정제화
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 불러오기
df = pd.read_csv('dialect_to_standard.csv')
dialect_sentences = df['방언'].values
standard_sentences = df['표준어'].values

# 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dialect_sentences + standard_sentences)
dialect_seq = tokenizer.texts_to_sequences(dialect_sentences)
standard_seq = tokenizer.texts_to_sequences(standard_sentences)

# 패딩 (길이 맞추기)
dialect_padded = pad_sequences(dialect_seq, padding='post')
standard_padded = pad_sequences(standard_seq, padding='post')


# 4. 기본 모델: Seq2Seq (Sequence to Sequence) 모델(입력→인코더→디코더→출력)
# Attention Mechanism 심화 모델.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# 하이퍼파라미터 설정
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
hidden_units = 512

# 인코더
encoder_inputs = Input(shape=(None,))
encoder_embedding = Dense(embedding_dim, activation='relu')(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 디코더
decoder_inputs = Input(shape=(None,))
decoder_embedding = Dense(embedding_dim, activation='relu')(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()


# 5. 모델 학습
model.fit([dialect_padded, standard_padded[:, :-1]], 
          np.expand_dims(standard_padded[:, 1:], -1),
          epochs=50,
          batch_size=32,
          validation_split=0.2)


# 6. 모델 평가(Bleu 평가)
from nltk.translate.bleu_score import sentence_bleu

reference = [['밥', '먹었니', '?']]
candidate = ['밥', '먹었니', '?']
print('BLEU Score:', sentence_bleu(reference, candidate))


# 7. 샘플 변환
def decode_sequence(input_seq):
    # 방언 입력을 표준어로 변환하는 코드 작성
    pass  # 구현은 모델 학습 후 구체적으로 설명

test_sentence = "밥 묵었나?"
print("변환 결과:", decode_sequence(test_sentence))