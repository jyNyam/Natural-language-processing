import torch
from transformers import BertTokenizer, EncoderDecoderModel

# 저장된 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderModel.from_pretrained("saved_model").to(device)
tokenizer = BertTokenizer.from_pretrained("saved_tokenizer")

# 변환 예측 함수
def translate_dialect(sentence):
    model.eval()
    input_encoding = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_encoding["input_ids"], 
            attention_mask=input_encoding["attention_mask"],
            max_length=128, 
            num_beams=5,  # Beam Search 적용
            early_stopping=True,
            num_return_sequences=1,  # 한 개의 결과만 반환
            decoder_start_token_id=tokenizer.cls_token_id  # 디코더 시작 토큰 설정
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 테스트 실행
test_sentences = ["밥 묵었나?", "어디 가노?", "뭐하노?", "그 사람 안 왔어?"]
for sentence in test_sentences:
    print(f"입력: {sentence} → 변환 결과: {translate_dialect(sentence)}")

# BLEU 점수 평가 함수
def evaluate_bleu(reference, candidate):
    ref_tokens = tokenizer.tokenize(reference)
    cand_tokens = tokenizer.tokenize(candidate)
    return sentence_bleu([ref_tokens], cand_tokens)

# BLEU 점수 테스트
references = ["밥 먹었니?"]
candidates = ["밥 묵었나?"]
bleu_score = evaluate_bleu(references[0], candidates[0])
print(f"BLEU Score: {bleu_score:.4f}")

# 전체 데이터셋을 사용한 BLEU 점수 계산 (데이터셋을 로드하고 번역 결과를 평가)
# total_bleu_score = 0
# for i in range(len(val_dialect)):
#     pred = translate_dialect(val_dialect[i])
#     total_bleu_score += evaluate_bleu(val_standard[i], pred)

# total_bleu_score /= len(val_dialect)
# print(f"Total BLEU Score: {total_bleu_score:.4f}")
