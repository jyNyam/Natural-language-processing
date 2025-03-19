# 한국어 방언-표준어 변환 모델

## Description
이 프로젝트는 한국어 방언을 표준어로 변환하는 Seq2Seq 모델을 구현합니다. Attention Mechanism을 포함하여 보다 정확한 변환을 목표로 합니다.  
본 프로젝트는 자연어 처리(NLP) 연구 및 인공지능 학습을 위한 **실험적 목적**을 갖습니다. 또한, `Korpora` 라이브러리를 활용하여 추가적인 데이터 증강이 가능합니다.


## 추후 보안 및 수정 방향
1. 데이터 활용을 구체화
Korpora.load("modu_web")을 사용하지만, 방언-표준어 쌍 데이터로 가공하는 과정이 부족해 보임. 데이터 정제 과정(불필요한 문자 제거, 구어체 정규화 등) 추가할 예정.

2. 모델 아키텍처 업그레이드
LSTM 기반 Seq2Seq 대신 Transformer 또는 T5 (Text-to-Text Transfer Transformer) 적용하여 BERT 기반 모델(KoBART, KoGPT, BERT2BERT)과 비교 분석이 필요함.

3. 성능 평가 지표 다양화
BLEU 외에도 Character Error Rate (CER), Translation Edit Rate (TER), Perplexity 등 다양한 평가가 필요함.

4. 모델 학습 & 결과 분석 추가
방언 종류별(지역에 따른 방언) 성능을 비교하여야 함.
학습 과정 시각화 (Loss 그래프, Attention weight 시각화) 자료를 제시하도록 해야 함.

## Getting Started

### Dependencies
* Python 3.8 이상
* TensorFlow
* scikit-learn
* NumPy
* pandas
* KoNLPy
* NLTK
* Korpora

설치 방법:
```bash
pip install -r requirements.txt
```

### Installing
1. 본 저장소를 클론합니다.
```bash
git clone https://github.com/your-repo/dialect-to-standard.git
cd dialect-to-standard
```
2. 필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### Executing program
1. 데이터셋을 생성합니다.
```python
python create_dataset.py
```
   - `create_dataset.py`는 방언-표준어 쌍 데이터를 생성하며, `modu_web` 코퍼스를 이용한 데이터 확장이 가능합니다.
   - 예시:
   ```python
   from Korpora import Korpora
   corpus = Korpora.load("modu_web")
   print(corpus.get_all_texts()[:5])  # 데이터 미리보기
   ```

2. 모델을 학습합니다.
```python
python train_model.py
```
   - Seq2Seq 모델을 학습시키며 Attention Mechanism을 포함합니다.

3. 변환 테스트를 실행합니다.
```python
python test_model.py
```
   - `decode_sequence()`를 통해 입력된 방언을 표준어로 변환합니다.

## Help
* 실행 중 오류가 발생할 경우 라이브러리 버전을 확인하세요.
```bash
pip list
```
* 데이터셋이 제대로 로드되지 않는다면, `create_dataset.py`의 파일 경로를 확인하세요.

## Authors
작성자: 허진영  
GitHub: [jyNyam](https://github.com/jyNyam)

## Version History
* 0.2
    * 모델 최적화 및 버그 수정
    * See [commit change]() or See [release history]()
* 0.1
    * 초기 릴리스



## * 수정 내역 * 
[1차] language.py 에서 아래와 같은 수정 내역을 통해 language_1st.py 로 재수정함.
[2차] language_1st.py 에서 아래와 같은 수정 내역을 통해 language_2nd.py 로 재수정함.
[3차] language_2nd.py 에서 아래와 같은 수정 내역을 통해 language_3rd 폴더로 재수정함.

# * [1차 수정 내역] *
# Embedding Layer 적용
기존 Dense(embedding_dim, activation='relu') → Embedding()으로 수정
Tokenizer로 만든 인덱스를 Embedding()을 통해 학습

# 입력 데이터 정제화
형태소 분석(Okt())을 적용하여 데이터 정제 후 Tokenizer에 입력
pad_sequences에서 maxlen을 자동으로 계산

# 디코딩 함수 구현 (decode_sequence())
입력 문장을 Tokenizer를 통해 변환 후 모델을 통해 예측
argmax()를 사용해 최적 단어 선택

# BLEU 평가 수정
candidate는 리스트 형태([['밥', '먹었니', '?']])로 변경

# 기대효과
오류 최소화
안정적인 Seq2Seq 모델 구현 지향
방언을 자연스러운 표준어로 변환



# * [2차 수정 내역] *
# Korpora.load("modu_web")으로 데이터를 불러온 형태
from Korpora import Korpora

corpus = Korpora.load("modu_web")
print(corpus)

# 데이터 불러오기 수정
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



# * [3차 수정 내역] *
### 1. LSTM 기반 Seq2Seq → BERT 기반 Encoder-Decoder
- 기존 LSTM 기반 Seq2Seq 모델을 **BERT 기반 Encoder-Decoder 모델**로 변경
- `tensorflow` 대신 `transformers`, `torch`를 활용하여 **사전 학습된 BERT 모델**을 사용

- BERT(Bidirectional Encoder Representation from Transforformer: 양방향 인코더로 표현하는 변환기): 사전에 학습된 Bert 모델을 단 하나의 레이어로 미세 조정하여 언어 추론과 같은 광범위한 작업을 위한 최신 모델을 만들 수 있으며, 특정 아키텍처를 수정하지 않아도 다양한 NLP에 쉽게 적응할 수 있는 성능이 우수한 모델이다.


- 주피터 노트북 설치한 후 실행
```bash
pip install notebook
jupyter notebook


### 2. 데이터 전처리 방식 변경
- **형태소 분석(Okt)**을 사용한 토큰화 방식 제거
- **BERT 토크나이저**(`BertTokenizer`)를 사용하여 문장을 토큰화
- `padding='post'` 방식에서 **BERT의 패딩 방식(`padding="max_length"`)으로 변경**

### 3. 모델 구조 변경
- 기존 LSTM Seq2Seq 모델에서 **BERT 기반 Encoder-Decoder 모델**
- BERT Encoder 사용 
- BERT Decoder 사용 
- 사전 학습된 BERT 토크나이저 사용 |
- `sparse_categorical_crossentropy` 를 `CrossEntropyLoss` 로 수정
- `pad_sequences`로 패딩 적용을 `BertTokenizer`의 `max_length` 로 수정

### 4. 학습 방식 변경
- `Adam` 옵티마이저 → **`AdamW`**로 변경 (BERT 최적화에 적합)
- Early Stopping 콜백 추가 (`EarlyStoppingCallback` 사용)
- PyTorch 기반 데이터 로더 (`Dataset`, `DataLoader`) 적용

### 5. 번역 및 BLEU 점수 평가 개선
- LSTM 모델의 **Greedy Decoding** 대신 **Beam Search** 적용
- `BLEU` 평가를 위한 `nltk` 활용

## 실행 방법
### 1. 환경 설정
```bash
pip install torch transformers Korpora nltk scikit-learn
```

### 2. 모델 학습 및 저장
```bash
python train.py
```

### 3. 테스트 실행
```bash
python test.py
```

### 4. 예제 테스트
```python
translate_dialect("밥 묵었나?")  # → "밥 먹었니?"
```

## BLEU 점수 평가
```python
from nltk.translate.bleu_score import sentence_bleu
references = [["밥", "먹었니", "?"]]
candidates = [["밥", "묵었나", "?"]]
bleu_score = sentence_bleu(references, candidates)
print(f"BLEU Score: {bleu_score:.4f}")
```

## 파일 구조
```
├── train.py  # 모델 학습 코드
├── test.py   # 변환 테스트 코드
├── README.md  # 설명서 (현재 파일)
├── saved_model/  # 학습된 모델 저장 폴더
├── saved_tokenizer/  # 학습된 토크나이저 저장 폴더
```

# 오류 일지
## 2025.03.02 오류내역
- ValueError: Not found corpus files. Check root_dir_or_paths 오류는 modu_web 코퍼스가 로컬에 다운로드되지 않았거나, Korpora가 해당 데이터를 찾지 못할 때 발생. 
- 해결책: modu_web Korpora download


## 2025.03.15 오류 내역
- 승인받은 pkg 파일은 말뭉치 데이터가 아닌 것을 확인하였으며, 국립국어원에서 말뭉치 데이터를 직접 다운로드하여 해제 후 Korpora로 로드해야 함.
- 다운로드받은 파일은 INNORIX 관련 실행 파일과 설정 파일만 있는 상황이며, 정상적인 다운로드가 이뤄지지 않았음을 확인함.
- 해결책: Mac이 아닌 Window PC에서 다운로드 받은 후 파일 이동한 후 다시 확인

## 2025.03.16 오류 내역
- 내려받은 파일이 텍스트가 아닌 음성(wav), 메타데이터(xlsx)인 것을 확인.
- 기존의 코드(1st, 2nd, 3rd)는 단순히 텍스트 쌍만을 데이터로 불러오게 한 코드였으므로 wav 파일과 xlsx 파일을 매칭해야 함.
- 해결책: 엑셀 파일을 load하고, 엑셀에 필요한 칼럼을 추출하여 wav 파일과 각각 데이터를 매칭해야 함.





## License
이 프로젝트는 **비상업적 용도로만 사용 가능**한 **Custom Non-Commercial License**를 따릅니다.  
즉, 본 소프트웨어는 **비영리적 목적**으로만 사용, 수정, 배포할 수 있습니다.  
상업적 용도로 사용하려면 별도의 허가가 필요합니다.  

자세한 사항은 [LICENSE](./LICENSE) 파일을 참조하세요.  

For commercial licensing inquiries, please contact: **[jyardent@gmail.com]**.

## Acknowledgments
본 프로젝트는 아래의 자료에서 영감을 받았습니다.
* [KoNLPy: 파이썬 한국어 NLP](https://konlpy.org/ko/latest/)
* [Korpora: Korean Corpora Archives](https://ko-nlp.github.io/Korpora/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch 공식 문서](https://pytorch.org/)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805?source=post_page)
* etc
```


