from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn WikiText passages into DeepSeek generation prompts.")
    parser.add_argument("--output", type=Path, default=Path("data/wikitext-prompts.jsonl"))
    parser.add_argument("--config", default="wikitext-103-raw-v1", choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records < 1:
        raise ValueError("--max-records must be at least 1")
    if args.min_chars < 1 or args.max_chars < args.min_chars:
        raise ValueError("Require 1 <= --min-chars <= --max-chars")

    parquet_url = (
        "hf://datasets/Salesforce/wikitext@refs/convert/parquet/"
        f"{args.config}/{args.split}/0000.parquet"
    )
    dataset = load_dataset("parquet", data_files={args.split: parquet_url}, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as target:
        for row in dataset:
            passage = " ".join(str(row["text"]).split())
            if len(passage) < args.min_chars:
                continue
            passage = passage[: args.max_chars]
            prompt = (
                "Using the source passage below, write a clear, accurate, self-contained explanation "
                "for a learner. Do not mention the source passage or this instruction.\n\n"
                f"Source passage:\n{passage}"
            )
            target.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.max_records:
                break
    print(f"Wrote {written} DeepSeek prompts to {args.output}")


if __name__ == "__main__":
    main()