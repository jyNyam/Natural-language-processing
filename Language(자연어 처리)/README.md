# 한국어 방언-표준어 변환 모델

## Description
이 프로젝트는 한국어 방언을 표준어로 변환하는 Seq2Seq 모델을 구현합니다. Attention Mechanism을 포함하여 보다 정확한 변환을 목표로 합니다.  
본 프로젝트는 자연어 처리(NLP) 연구 및 인공지능 학습을 위한 **실험적 목적**을 갖습니다. 또한, `Korpora` 라이브러리를 활용하여 추가적인 데이터 증강이 가능합니다.

## Getting Started
[1차] language.py 에서 아래와 같은 수정 내역을 통해 language_1st.py 로 재수정함.

[2차] language_1st.py 에서 아래와 같은 수정 내역을 통해 language_2nd.py 로 재수정함.

[3차] language_2nd.py 에서 아래와 같은 수정 내역을 통해 language_3rd 폴더로 재수정함.


# [1차 수정 내역]
## Embedding Layer 적용
기존 Dense(embedding_dim, activation='relu') → Embedding()으로 수정
Tokenizer로 만든 인덱스를 Embedding()을 통해 학습

## 입력 데이터 정제화
형태소 분석(Okt())을 적용하여 데이터 정제 후 Tokenizer에 입력
pad_sequences에서 maxlen을 자동으로 계산

## 디코딩 함수 구현 (decode_sequence())
입력 문장을 Tokenizer를 통해 변환 후 모델을 통해 예측
argmax()를 사용해 최적 단어 선택

## BLEU 평가 수정
candidate는 리스트 형태([['밥', '먹었니', '?']])로 변경

## 기대효과
오류 최소화
안정적인 Seq2Seq 모델 구현 지향
방언을 자연스러운 표준어로 변환



# [2차 수정 내역]
## Korpora.load("modu_web")으로 데이터를 불러온 형태
from Korpora import Korpora

corpus = Korpora.load("modu_web")
print(corpus)

## 데이터 불러오기 수정
[기존 코드]

df = pd.read_csv('dialect_to_standard.csv')
dialect_sentences = df['방언'].values
standard_sentences = df['표준어'].values

[수정 코드] # 방언과 표준어 데이터 분리

from Korpora import Korpora

corpus = Korpora.load("modu_web")

dialect_sentences = [pair[0] for pair in corpus.pairs]  # source(방언)
standard_sentences = [pair[1] for pair in corpus.pairs]  # target(표준어)


/(이유)/ Korpora.load("modu_web")은 (source, target) 쌍으로 제공되므로 .pairs 속성을 활용해야 합니다.
기존의 CSV 기반 데이터 로딩이 필요 없으며, 직접 pairs에서 데이터를 가져오도록 수정합니다.

## 데이터 전처리 부분 수정
[기존 코드]

dialect_sentences = tokenize_korean(df['방언'].values)
standard_sentences = tokenize_korean(df['표준어'].values)

[수정 코드]

dialect_sentences = tokenize_korean(dialect_sentences)
standard_sentences = tokenize_korean(standard_sentences)


/(이유)/
dialect_sentences와 standard_sentences는 리스트 형태이므로 바로 tokenize_korean()에 전달하게 함으로 기존의 df['방언'].values처럼 pandas를 사용할 필요가 없습니다.


## 토큰화 및 패딩 조정
[기존 코드]

tokenizer.fit_on_texts(dialect_sentences + standard_sentences)
dialect_seq = tokenizer.texts_to_sequences(dialect_sentences)
standard_seq = tokenizer.texts_to_sequences(standard_sentences)

max_len = max(max(len(seq) for seq in dialect_seq), max(len(seq) for seq in standard_seq))
dialect_padded = pad_sequences(dialect_seq, maxlen=max_len, padding='post')
standard_padded = pad_sequences(standard_seq, maxlen=max_len, padding='post')

[수정 코드] # 데이터가 방대하므로 일부 샘플(예: 50,000개)만 사용 가능

num_samples = 50000  
dialect_sentences = dialect_sentences[:num_samples]
standard_sentences = standard_sentences[:num_samples]

tokenizer.fit_on_texts(dialect_sentences + standard_sentences)

dialect_seq = tokenizer.texts_to_sequences(dialect_sentences)
standard_seq = tokenizer.texts_to_sequences(standard_sentences)

max_len = 30  # 너무 길면 모델 학습이 어려우므로 적절한 길이 설정
dialect_padded = pad_sequences(dialect_seq, maxlen=max_len, padding='post')
standard_padded = pad_sequences(standard_seq, maxlen=max_len, padding='post')


/(이유)/
MODU Web 데이터는 약 20만 개의 문장쌍으로 데이터가 크기 때문에, 학습 속도와 메모리 문제를 고려하여 일부 샘플만 사용할 수 있도록 num_samples = 50000을 설정하였습니다. max_len을 임의로 30으로 설정하여 긴 문장은 적절히 설정값에 따라 자릅니다.


## 모델 학습 데이터 수정
[기존 코드]

model.fit([dialect_padded, standard_padded[:, :-1]], 
          np.expand_dims(standard_padded[:, 1:], -1),
          epochs=50,
          batch_size=32,
          validation_split=0.2)

[수정 코드]

model.fit([dialect_padded, standard_padded[:, :-1]], 
          np.expand_dims(standard_padded[:, 1:], -1),
          epochs=10,  # 데이터가 많으므로 우선 10 epoch만 학습
          batch_size=64,  # 더 큰 배치 크기 사용 가능
          validation_split=0.1)  # 검증 데이터 10% 사용


/(이유)/
데이터가 많으므로 훈련 데이터를 50 epochs로 학습하면 과적합 및 학습 시간이 오래 걸릴 수 있습니다. 따라서 배치 크기를 64로 늘려서 더 빠르게 학습하도록 조정했습니다. validation_split=0.1로 설정하여 10%를 검증 데이터로 사용합니다.


## 테스트 및 예측 함수 적용 수정
[기존 코드]

test_sentence = "밥 묵었나?"
print("변환 결과:", decode_sequence(test_sentence))

[수정 코드]

test_sentences = ["밥 묵었나?", "어디 가노?", "뭐하노?", "그 사람 안 왔어?"]

for sentence in test_sentences:
    print(f"입력: {sentence} → 변환 결과: {decode_sequence(sentence)}")


/(이유)/
다양한 입력 문장에 대해 변환 결과를 확인할 수 있도록 여러 개의 테스트 문장을 추가했습니다.



# [3차 수정 내역]
** Languagu.3rd 참고 **


# 결론
1. Korpora.load("modu_web")을 사용하면 기존 CSV 파일을 직접 다룰 필요 없이 더 방대한 데이터로 학습할 수 있습니다. 하지만 데이터 크기가 크므로, 적절한 샘플링 및 학습 속도 최적화를 할 예정입니다.
2. 기존 코드에서 데이터 로딩, 전처리, 학습 부분을 조금씩 수정 및 보안을 할 예정입니다.

