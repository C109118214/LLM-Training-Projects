from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("./models")  # 載入已訓練模型

training_args = TrainingArguments(
    output_dir="./models_finetune",
    per_device_train_batch_size=2,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None  # 這裡放你的專屬數據集
)

trainer.train()
