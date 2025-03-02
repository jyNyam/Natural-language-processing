# 방언 → 표준어 변환 모델 (Seq2Seq vs. BERT 기반)

## 개요
이 프로젝트는 한국어 방언 문장을 표준어로 변환하는 모델을 구현합니다. 기존 LSTM 기반의 Seq2Seq 모델을 BERT 기반 Encoder-Decoder 모델로 개선하여 성능을 향상시켰습니다.

## 변경 사항 요약
### 1. LSTM 기반 Seq2Seq → BERT 기반 Encoder-Decoder
- 기존 LSTM 기반 Seq2Seq 모델을 **BERT 기반 Encoder-Decoder 모델**로 변경
- `tensorflow` 대신 `transformers`, `torch`를 활용하여 **사전 학습된 BERT 모델**을 사용

- 주피터 노트북 설치한 후 실행
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
- 
