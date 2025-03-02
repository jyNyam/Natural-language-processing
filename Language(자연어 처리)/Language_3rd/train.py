from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
import torch
from datasets import load_dataset

# 데이터 로드
dataset = load_dataset("kor_text", split="train")
train_texts, test_texts = dataset["train"], dataset["test"]

# 토크나이저 설정
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
train_encodings = tokenizer(train_texts, padding="max_length", truncation=True)
test_encodings = tokenizer(test_texts, padding="max_length", truncation=True)

# 모델 설정
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased")

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=test_encodings
)

# 학습 실행
trainer.train()

# 모델 저장
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_tokenizer")
