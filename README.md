# 한국어 방언-표준어 변환 모델

## Description
이 프로젝트는 한국어 방언을 표준어로 변환하는 Seq2Seq 모델을 구현합니다. Attention Mechanism을 포함하여 보다 정확한 변환을 목표로 합니다.  
본 프로젝트는 자연어 처리(NLP) 연구 및 인공지능 학습을 위한 **실험적 목적**을 갖습니다. 또한, `Korpora` 라이브러리를 활용하여 추가적인 데이터 증강이 가능합니다.

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
* etc
```
