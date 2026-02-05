from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
)
from src.config import SETTINGS
from src.dataset import build_hf_dataset

def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=SETTINGS.max_input_len,
        truncation=True
    )
    labels = tokenizer(
        examples["target"],
        max_length=SETTINGS.max_output_len,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    SETTINGS.out_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.model_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(SETTINGS.data_dir / "recipes_train.jsonl")
    val_path = str(SETTINGS.data_dir / "recipes_val.jsonl")

    train_ds = build_hf_dataset(train_path)
    val_ds = build_hf_dataset(val_path)

    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(SETTINGS.base_model)

    train_tok = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = TrainingArguments(
        output_dir=str(SETTINGS.out_dir / "train_runs"),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-4,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        fp16=False,   # CPU friendly
        report_to="none"
    )

    trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator
            )


    trainer.train()

    model.save_pretrained(str(SETTINGS.model_dir))
    tokenizer.save_pretrained(str(SETTINGS.model_dir))
    print(f"\nSaved fine-tuned model to: {SETTINGS.model_dir}")

if __name__ == "__main__":
    main()
