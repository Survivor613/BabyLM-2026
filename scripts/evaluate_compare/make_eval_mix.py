from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_lines(path: Path, max_per_file: int) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= max_per_file:
                break
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed held-out eval text file from BabyLM26 corpus files.")
    parser.add_argument("--source-dir", type=Path, default=Path("reference/babylm26-nlp-spring/new_data2"))
    parser.add_argument("--output", type=Path, default=Path("data/babylm26_eval_mix.txt"))
    parser.add_argument("--max-per-file", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(args.source_dir.glob("*.train.txt"))
    if not files:
        raise FileNotFoundError(f"No *.train.txt files found in {args.source_dir}")

    mixed = []
    for path in files:
        for line in read_lines(path, args.max_per_file):
            mixed.append((path.name, line))

    random.Random(args.seed).shuffle(mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(line for _, line in mixed) + "\n", encoding="utf-8")

    print(f"source_dir={args.source_dir}")
    print(f"files={len(files)}")
    print(f"examples={len(mixed)}")
    print(f"output={args.output}")
    print("first_examples:")
    for name, line in mixed[:5]:
        print(f"[{name}] {line[:160]}")


if __name__ == "__main__":
    main()
