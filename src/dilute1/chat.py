from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .runtime import select_device
except ImportError:
    from runtime import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompts through a trained Dilute-1 checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/dilute-1", help="Saved Dilute-1 checkpoint directory.")
    parser.add_argument("--prompt", help="Prompt to run. Omit it to enter interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    return parser.parse_args()


def generate(model, tokenizer, prompt: str, device: torch.device, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generation_args.update({"temperature": args.temperature, "top_p": args.top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    if args.prompt is not None:
        print(generate(model, tokenizer, args.prompt, device, args))
        return

    print("Dilute-1 interactive mode. Type /exit to stop.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except EOFError:
            break
        if prompt == "/exit":
            break
        if not prompt:
            continue
        print(f"Dilute-1: {generate(model, tokenizer, prompt, device, args)}")


if __name__ == "__main__":
    main()
