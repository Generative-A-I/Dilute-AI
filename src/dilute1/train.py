from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    from .runtime import precision_flags, select_device
except ImportError:
    from runtime import precision_flags, select_device


class TrainingProgressCallback(TrainerCallback):
    def __init__(self) -> None:
        self.started_at = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        print(f"Training started: {state.max_steps:,} optimizer steps planned.", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        total_steps = max(state.max_steps, 1)
        completed_steps = min(state.global_step, total_steps)
        percent = completed_steps / total_steps * 100
        elapsed = time.monotonic() - self.started_at
        loss = logs.get("loss")
        loss_text = f" | loss {loss:.4f}" if isinstance(loss, float) else ""
        print(
            f"Training progress: {completed_steps:,}/{total_steps:,} "
            f"({percent:5.1f}%) | epoch {state.epoch or 0:.2f} | "
            f"elapsed {elapsed / 60:.1f} min{loss_text}",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        print("Training finished: model checkpoint is being saved.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dilute-1 from scratch on teacher-generated JSONL data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer ID or local tokenizer path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dilute-1"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "no", "bf16", "fp16"], default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Preparing Dilute-1 training on {device}.", flush=True)
    use_bf16, use_fp16 = precision_flags(device, args.precision)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    print(f"Loaded {len(dataset):,} training examples.", flush=True)

    def format_example(example: dict[str, str]) -> dict[str, str]:
        return {"text": f"### Prompt\n{example['prompt']}\n### Response\n{example['completion']}"}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.max_length,
        n_ctx=args.max_length,
        n_embd=args.hidden_size,
        n_layer=args.layers,
        n_head=args.heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        dataloader_pin_memory=device.type == "cuda",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[TrainingProgressCallback()],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {"device": str(device), "parameters": sum(parameter.numel() for parameter in model.parameters())}
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
