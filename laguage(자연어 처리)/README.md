# portpolios


# (비상업적 연구 목적으로만 사용 가능)

# 방언 → 표준어 변환 모델 (Dialect-to-Standard Conversion)

## 프로젝트 개요
이 프로젝트는 Seq2Seq 모델을 이용하여 한국어 방언을 표준어로 변환하는 모델을 개발하는 것입니다.


## 기술 스택
- Python, TensorFlow, Keras
- NLP (KoNLPy, Okt 형태소 분석기)
- LSTM 기반 Seq2Seq 모델

## 데이터셋
- 수집한 방언-표준어 데이터 (dialect_to_standard.csv)

## 실행 방법
1. 필요한 라이브러리 설치
   ```bash
   pip install -r requirements.txt


# (추후 수정) 모두의 말뭉치
from Korpora import Korpora
corpus = Korpora.load("modu_web")


