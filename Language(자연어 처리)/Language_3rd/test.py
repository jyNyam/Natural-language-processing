from transformers import BertTokenizer, EncoderDecoderModel
import torch
from nltk.translate.bleu_score import sentence_bleu

# 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("./saved_tokenizer")
model = EncoderDecoderModel.from_pretrained("./saved_model")

def translate_dialect(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 테스트 실행
input_text = "밥 묵었나?"
output_text = translate_dialect(input_text)
print(f"입력: {input_text} → 변환: {output_text}")

# BLEU 점수 평가
references = [["밥", "먹었니", "?"]]
candidates = [output_text.split()]
bleu_score = sentence_bleu(references, candidates)
print(f"BLEU Score: {bleu_score:.4f}")
