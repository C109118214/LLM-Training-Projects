from transformers import Trainer, TrainingArguments
import deepspeed

training_args = TrainingArguments(
    output_dir="./models_deepspeed",
    per_device_train_batch_size=4,
    deepspeed="./ds_config.json"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None
)

trainer.train()

