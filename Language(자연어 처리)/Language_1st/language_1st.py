# 필요한 라이브러리 설치
# pip install konlpy pandas numpy scikit-learn tensorflow nltk

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from konlpy.tag import Okt
from nltk.translate.bleu_score import sentence_bleu

# 1. 데이터셋 생성 및 저장
data = {
    '방언': ['밥 묵었나?', '어디 가노?', '뭐하노?'],
    '표준어': ['밥 먹었니?', '어디 가니?', '뭐 하니?']
}
df = pd.DataFrame(data)
df.to_csv('dialect_to_standard.csv', index=False)

# 2. 데이터 전처리 및 형태소 분석
okt = Okt()

def tokenize_korean(sentences):
    return [" ".join(okt.morphs(sentence)) for sentence in sentences]

# 데이터 불러오기
df = pd.read_csv('dialect_to_standard.csv')
dialect_sentences = tokenize_korean(df['방언'].values)
standard_sentences = tokenize_korean(df['표준어'].values)

# 3. 토큰화 및 정제화
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(dialect_sentences + standard_sentences)

dialect_seq = tokenizer.texts_to_sequences(dialect_sentences)
standard_seq = tokenizer.texts_to_sequences(standard_sentences)

max_len = max(max(len(seq) for seq in dialect_seq), max(len(seq) for seq in standard_seq))
dialect_padded = pad_sequences(dialect_seq, maxlen=max_len, padding='post')
standard_padded = pad_sequences(standard_seq, maxlen=max_len, padding='post')

# 4. Seq2Seq 모델 구현
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
hidden_units = 512

# 인코더
encoder_inputs = Input(shape=(max_len,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 디코더
decoder_inputs = Input(shape=(max_len,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
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

# 6. BLEU 평가
reference = [['밥', '먹었니', '?']]
candidate = ['밥', '먹었니', '?']
print('BLEU Score:', sentence_bleu(reference, [candidate]))

# 7. 변환 함수 구현
def decode_sequence(input_text):
    input_seq = tokenizer.texts_to_sequences([tokenize_korean([input_text])[0]])
    input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')

    states_value = model.predict([input_padded, np.zeros((1, max_len))])
    decoded_sentence = []
    
    for word_idx in states_value[0].argmax(axis=-1):
        if word_idx == 0:
            break
        decoded_sentence.append(tokenizer.index_word[word_idx])

    return " ".join(decoded_sentence)

# 테스트 실행
test_sentence = "밥 묵었나?"
print("변환 결과:", decode_sequence(test_sentence))
