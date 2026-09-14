from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .runtime import select_device
except ImportError:
    from runtime import select_device


DEFAULT_TEACHER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Dilute-1 training data with a DeepSeek teacher.")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER, help="Hugging Face model ID or local teacher path.")
    parser.add_argument("--prompts", type=Path, default=Path("data/wikitext-prompts.jsonl"), help="JSONL with one object containing a 'prompt' field per line.")
    parser.add_argument("--output", type=Path, default=Path("data/teacher.jsonl"), help="Output JSONL containing prompt and completion.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.prompts.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{args.prompts}:{line_number} must contain a non-empty string 'prompt'.")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generation_args = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.temperature > 0:
                generation_args.update({"temperature": args.temperature, "top_p": args.top_p})
            with torch.inference_mode():
                output = teacher.generate(**inputs, **generation_args)
            completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            target.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
