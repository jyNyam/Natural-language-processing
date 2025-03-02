import torch
from transformers import BertTokenizer, EncoderDecoderModel
from Korpora import Korpora
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. 데이터 불러오기
corpus_dir = "./data/modu_web"  # 다운로드 받은 코퍼스 파일의 경로
corpus = Korpora.load("modu_web", root_dir=corpus_dir)
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

# 모델 저장
save_path = "saved_model"
tokenizer_save_path = "saved_tokenizer"

model.save_pretrained(save_path)
tokenizer.save_pretrained(tokenizer_save_path)
