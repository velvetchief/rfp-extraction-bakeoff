"""Turn the 11 teacher matrices into chunk-level training pairs.

Each pair is one conversation: the same system prompt and user turn the
baselines saw (minus the worked example — the fine-tune replaces it), with the
teacher's rows for that chunk as the assistant answer. Chunks reuse
baseline.chunk_pages at 4500 words so training and tuned inference see the
same distribution; rows are assigned to chunks by their page number.

Usage: python3 scripts/make_train_pairs.py [src_dir out_file]
       python3 scripts/make_train_pairs.py                # round 1 (default)
       python3 scripts/make_train_pairs.py data/train2 data/pairs/train_v2.jsonl
Writes: one {"rfp_id", "chunk", "messages"} JSON line per chunk
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baseline import CHUNK_WORDS, chunk_pages, rfp_text, user_turn  # noqa: E402
from prompts import EXTRACTION_SYSTEM  # noqa: E402

CHUNK = CHUNK_WORDS["qwen"]  # one size for every contestant's training data
HELD_OUT = {"3202600219", "HHS0015558v2", "3202600177"}


def chunk_page_numbers(chunk: str) -> set[int]:
    return {int(m) for m in re.findall(r"^\[page (\d+)\]", chunk, re.M)}


def answer_text(rfp_id: str, rows: list[dict]) -> str:
    """Compact matrix JSON, one row per line (same shape as the few-shot)."""
    if not rows:
        return json.dumps({"rfp_id": rfp_id, "requirements": []})
    body = ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    return f'{{"rfp_id": "{rfp_id}", "requirements": [\n{body}\n]}}'


def main() -> None:
    if len(sys.argv) == 3:
        src_dir = ROOT / sys.argv[1]
        out_path = ROOT / sys.argv[2]
    elif len(sys.argv) == 1:
        src_dir = ROOT / "data" / "train"
        out_path = ROOT / "data" / "pairs" / "train.jsonl"
    else:
        sys.exit(__doc__)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    dropped = 0
    matrix_paths = sorted(src_dir.glob("*.json"))
    if not matrix_paths:
        sys.exit(f"no matrices in {src_dir}")
    for matrix_path in matrix_paths:
        rfp_id = matrix_path.stem
        assert rfp_id not in HELD_OUT, f"{rfp_id} must stay held out"
        matrix = json.loads(matrix_path.read_text())
        rows = matrix["requirements"]
        chunks = chunk_pages(rfp_text(rfp_id), CHUNK)
        covered: set[str] = set()
        for n, chunk in enumerate(chunks, 1):
            pages = chunk_page_numbers(chunk)
            mine = [r for r in rows if r["page"] in pages]
            covered.update(r["id"] for r in mine)
            renumbered = [dict(r, id=f"R-{i:03d}")
                          for i, r in enumerate(mine, 1)]
            pairs.append({
                "rfp_id": rfp_id,
                "chunk": f"{n}/{len(chunks)}",
                "n_rows": len(renumbered),
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_turn(rfp_id, chunk)},
                    {"role": "assistant",
                     "content": answer_text(rfp_id, renumbered)},
                ],
            })
        missed = [r["id"] for r in rows if r["id"] not in covered]
        dropped += len(missed)
        tag = f"  MISSED {missed}" if missed else ""
        print(f"{rfp_id}: {len(chunks)} chunks, {len(rows)} rows{tag}")

    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    empty = sum(1 for p in pairs if p["n_rows"] == 0)
    print(f"\n{len(pairs)} pairs ({empty} with zero rows), "
          f"{dropped} teacher rows unassigned -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
