from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TEACHER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FALLBACK_TEACHER = "HuggingFaceTB/SmolLM2-135M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WikiText preparation, DeepSeek synthesis, tokenizer training, and Dilute-1 training."
    )
    parser.add_argument(
        "--teacher-model",
        default="auto",
        help=f"Teacher model ID/path, or auto (default: {DEFAULT_TEACHER}; low-disk fallback: {FALLBACK_TEACHER}).",
    )
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "no", "bf16", "fp16"], default="auto")
    parser.add_argument("--force", action="store_true", help="Regenerate existing intermediate files.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def choose_teacher(requested: str) -> str:
    if requested != "auto":
        return requested
    free_bytes = shutil.disk_usage(Path.cwd()).free
    deepseek_requires = 5 * 1024**3
    if free_bytes >= deepseek_requires:
        return DEFAULT_TEACHER
    print(
        f"Only {free_bytes / 1024**3:.1f} GB is free; using {FALLBACK_TEACHER} "
        f"instead of the {DEFAULT_TEACHER} checkpoint.",
        flush=True,
    )
    return FALLBACK_TEACHER


def main() -> None:
    args = parse_args()
    if args.records < 1:
        raise ValueError("--records must be at least 1")

    data_dir = Path("data")
    prompt_file = data_dir / "wikitext-prompts.jsonl"
    teacher_file = data_dir / "teacher.jsonl"
    tokenizer_dir = args.output_root / "tokenizer"
    model_dir = args.output_root / "dilute-1"
    python = sys.executable
    teacher_model = choose_teacher(args.teacher_model)

    if args.force or not prompt_file.exists():
        run([python, "-m", "dilute1.prepare_wikitext", "--output", str(prompt_file), "--max-records", str(args.records)])
    if args.force or not teacher_file.exists():
        run(
            [
                python,
                "-m",
                "dilute1.generate",
                "--teacher-model",
                teacher_model,
                "--prompts",
                str(prompt_file),
                "--output",
                str(teacher_file),
                "--max-new-tokens",
                "128",
                "--device",
                args.device,
            ]
        )
    if args.force or not (tokenizer_dir / "tokenizer.json").exists():
        run(
            [
                python,
                "-m",
                "dilute1.tokenizer",
                "--input",
                str(prompt_file),
                str(teacher_file),
                "--output-dir",
                str(tokenizer_dir),
                "--vocab-size",
                "32000",
            ]
        )
    run(
        [
            python,
            "-m",
            "dilute1.train",
            "--data",
            str(teacher_file),
            "--tokenizer",
            str(tokenizer_dir),
            "--output-dir",
            str(model_dir),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--max-length",
            "512",
            "--layers",
            "6",
            "--hidden-size",
            "384",
            "--heads",
            "6",
            "--batch-size",
            "1",
            "--gradient-accumulation",
            "8",
        ]
    )
    print(f"\nDilute-1 training completed. Model saved to {model_dir}")


if __name__ == "__main__":
    main()