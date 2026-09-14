from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Dilute-1-compatible tokenizer from local text.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Text or JSONL training files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--format", choices=["auto", "text", "jsonl"], default="auto")
    return parser.parse_args()


def iter_training_text(paths: list[Path], file_format: str) -> Iterator[str]:
    for path in paths:
        detected_format = file_format
        if detected_format == "auto":
            detected_format = "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "text"
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                if detected_format == "jsonl":
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                    prompt = record.get("prompt", "")
                    completion = record.get("completion")
                    if not isinstance(prompt, str) or (completion is not None and not isinstance(completion, str)):
                        raise ValueError(f"{path}:{line_number} needs a string prompt and optional string completion")
                    if completion is None:
                        yield prompt
                    else:
                        yield f"### Prompt\n{prompt}\n### Response\n{completion}"
                else:
                    yield line


def main() -> None:
    args = parse_args()
    if args.vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"--vocab-size must be greater than {len(SPECIAL_TOKENS)}")
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be at least 1")

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train_from_iterator(
        iter_training_text(args.input, args.format),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(args.output_dir))
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        model_max_length=2048,
    )
    fast_tokenizer.save_pretrained(args.output_dir)
    print(f"Saved tokenizer with {len(fast_tokenizer)} tokens to {args.output_dir}")


if __name__ == "__main__":
    main()