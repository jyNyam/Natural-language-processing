import torch
from transformers import BertTokenizer, EncoderDecoderModel
from Korpora import Korpora
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu

# 1. 데이터 불러오기
corpus = Korpora.load("modu_web")
num_samples = 50000  # 사용할 샘플 수 조절

dialect_sentences = [pair[0] for pair in corpus.pairs[:num_samples]]  # 방언
standard_sentences = [pair[1] for pair in corpus.pairs[:num_samples]]  # 표준어

# 2. BERT 토크나이저 설정
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# 3. 데이터셋 클래스 정의
class DialectDataset(Dataset):
    def __init__(self, dialect_sentences, standard_sentences, tokenizer, max_len=128):
        self.dialect_sentences = dialect_sentences
        self.standard_sentences = standard_sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dialect_sentences)

    def __getitem__(self, idx):
        dialect = self.dialect_sentences[idx]
        standard = self.standard_sentences[idx]

        encoding = self.tokenizer(
            dialect,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            standard,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

# 4. 데이터셋 및 DataLoader 생성
max_len = 128  # BERT 모델의 최대 길이 설정
train_dialect, val_dialect, train_standard, val_standard = train_test_split(
    dialect_sentences, standard_sentences, test_size=0.1, random_state=42
)

train_dataset = DialectDataset(train_dialect, train_standard, tokenizer, max_len)
val_dataset = DialectDataset(val_dialect, val_standard, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 5. BERT 기반 Encoder-Decoder 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased").to(device)

# 6. 학습 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 7. Early Stopping 직접 구현
early_stopping_patience = 3
best_loss = float("inf")
epochs_no_improve = 0
epochs = 10  # 최대 학습 epoch 설정

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

    # Early Stopping 적용
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print("Early stopping")
        break

# 8. 변환 예측 함수
def translate_dialect(sentence):
    model.eval()
    input_encoding = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_encoding["input_ids"], 
            attention_mask=input_encoding["attention_mask"],
            max_length=max_len, 
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

# 모델 저장
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_tokenizer")

# 저장된 모델 로드
model = EncoderDecoderModel.from_pretrained("saved_model").to(device)
tokenizer = BertTokenizer.from_pretrained("saved_tokenizer")

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
