# 방언 → 표준어 변환 모델 (Seq2Seq vs. BERT 기반)

## 개요
이 프로젝트는 한국어 방언 문장을 표준어로 변환하는 모델을 구현합니다. 기존 LSTM 기반의 Seq2Seq 모델을 BERT 기반 Encoder-Decoder 모델로 개선하여 성능을 향상시켰습니다.

## 변경 사항 요약
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

## 참고 자료
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch 공식 문서](https://pytorch.org/)


## 2025.03.02 1차 오류내역
- ValueError: Not found corpus files. Check root_dir_or_paths 오류는 modu_web 코퍼스가 로컬에 다운로드되지 않았거나, Korpora가 해당 데이터를 찾지 못할 때 발생. 
- 해결책: modu_web Korpora download


## 2025.03.15 2차 오류 내역
- 승인받은 pkg 파일은 말뭉치 데이터가 아닌 것을 확인하였으며, 국립국어원에서 말뭉치 데이터를 직접 다운로드하여 해제 후 Korpora로 로드해야 함.
- 다운로드받은 파일은 INNORIX 관련 실행 파일과 설정 파일만 있는 상황이며, 정상적인 다운로드가 이뤄지지 않았음을 확인함.
- 해결책: Mac이 아닌 Window PC에서 다운로드 받은 후 파일 이동한 후 다시 확인