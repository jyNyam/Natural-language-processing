import numpy as np
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from nltk.translate.bleu_score import sentence_bleu

# 1. 데이터 불러오기
corpus = Korpora.load("modu_web")
num_samples = 50000  # 너무 크면 메모리 문제 발생, 일부 샘플만 사용

dialect_sentences = [pair[0] for pair in corpus.pairs[:num_samples]]  # 방언
standard_sentences = [pair[1] for pair in corpus.pairs[:num_samples]]  # 표준어

# 2. 형태소 분석 및 토큰화
def tokenize_korean(sentences):
    okt = Okt()
    return [okt.morphs(sentence) for sentence in sentences]

dialect_tokens = tokenize_korean(dialect_sentences)
standard_tokens = tokenize_korean(standard_sentences)

# 3. 토큰화 및 패딩
all_sentences = dialect_tokens + standard_tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_sentences)

# 시퀀스로 변환
dialect_seq = tokenizer.texts_to_sequences(dialect_tokens)
standard_seq = tokenizer.texts_to_sequences(standard_tokens)

# 최대 길이 설정 및 패딩
max_len = 30
dialect_padded = pad_sequences(dialect_seq, maxlen=max_len, padding='post')
standard_padded = pad_sequences(standard_seq, maxlen=max_len, padding='post')

# 4. Seq2Seq 모델 구성
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
epochs = 10  # 학습 횟수 조정
batch_size = 64

model.fit([dialect_padded, standard_padded[:, :-1]],
          np.expand_dims(standard_padded[:, 1:], -1),
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1)

# 6. BLEU 평가 함수
def evaluate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

# 7. 변환 예측 함수 (미완성, 학습 완료 후 적용 필요)
def decode_sequence(input_seq):
    # 여기에 변환 로직 추가 (Beam Search 또는 Greedy Decoding)
    return "변환 결과 (임시)"

# 테스트 실행
test_sentences = ["밥 묵었나?", "어디 가노?", "뭐하노?", "그 사람 안 왔어?"]
for sentence in test_sentences:
    print(f"입력: {sentence} → 변환 결과: {decode_sequence(sentence)}")
