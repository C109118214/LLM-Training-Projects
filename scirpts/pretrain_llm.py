from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 下載 GPT-2
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None  # 這裡要放你的數據集
)

trainer.train()

