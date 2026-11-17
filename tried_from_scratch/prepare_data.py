"""
Step 1 - Data: build corpus.txt for your tiny LLM.

Mixes two kinds of text:
  1. General English from TinyStories (so the model learns grammar)
  2. Your own text from the my_text/ folder (so it learns about you),
     repeated a few times so it isn't drowned out

Usage:
  pip install datasets
  python prepare_data.py

Output:
  corpus.txt       -> feed this to the tokenizer step
"""

import random
import re
from pathlib import Path

from datasets import load_dataset

# ---- settings -------------------------------------------------------------
GENERAL_MB = 20          # how much TinyStories text to take (MB)
MY_TEXT_DIR = Path("txt")
MY_TEXT_REPEAT = 1       # repeat your own text so the model sees it more
OUT_FILE = Path("corpus.txt")
SEED = 42
# --------------------------------------------------------------------------


def clean(text: str) -> str:
    """Basic cleanup: strip HTML tags, collapse extra blank lines and spaces."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_general(max_bytes: int) -> list[str]:
    """Stream TinyStories and stop once we have about max_bytes of text."""
    print(f"Downloading ~{max_bytes // 1_000_000} MB of TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    docs, total = [], 0
    for row in ds:
        story = clean(row["text"])
        if not story:
            continue
        docs.append(story)
        total += len(story.encode("utf-8"))
        if total >= max_bytes:
            break
    print(f"  got {len(docs):,} stories ({total / 1e6:.1f} MB)")
    return docs


def load_my_text(folder: Path) -> list[str]:
    """Read every .txt / .md file, split on <eos> so each conversation is its own document."""
    if not folder.exists():
        print(f"No {folder}/ folder found - create it and add your .txt/.md files.")
        return []
    docs = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() in {".txt", ".md"}:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            for part in raw.split("<eos>"):   # split FIRST, while <eos> still exists
                part = clean(part)            # then clean each piece
                if part:
                    docs.append(part)
    size = sum(len(d.encode("utf-8")) for d in docs)
    print(f"  your text: {len(docs):,} docs ({size / 1e3:.1f} KB)")
    return docs

def main() -> None:
    random.seed(SEED)
    general = load_general(GENERAL_MB * 1_000_000)
    mine = load_my_text(MY_TEXT_DIR) * MY_TEXT_REPEAT

    docs = general + mine
    random.shuffle(docs)  # mix them so each training batch sees both kinds

    # <eos> separates documents; the tokenizer step registers it as a special token
    OUT_FILE.write_text("\n<eos>\n".join(docs), encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB, {len(docs):,} docs)")


if __name__ == "__main__":
    main()