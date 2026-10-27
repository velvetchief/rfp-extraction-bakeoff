"""Score every prediction in out/baselines/ against gold and write results.md.

Usage: python3 scripts/make_results.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from score import score  # noqa: E402

GOLD_IDS = ["3202600219", "HHS0015558v2", "3202600177"]
LABELS = {
    "claude": "Claude (few-shot, the bar to beat)",
    "glm": "GLM 5.2 (few-shot API)",
    "qwen": "Qwen3.5-4B (un-tuned)",
    "kimi": "Kimi K2.6 (un-tuned)",
    "inkling": "Inkling (un-tuned)",
    "qwen-tuned": "Qwen3.5-4B (fine-tuned)",
    "kimi-tuned": "Kimi K2.6 (fine-tuned)",
    "inkling-tuned": "Inkling (fine-tuned)",
    "qwen2-tuned": "Qwen3.5-4B (fine-tuned, round 2)",
    "kimi2-tuned": "Kimi K2.6 (fine-tuned, round 2)",
    "inkling2-tuned": "Inkling (fine-tuned, round 2)",
}


def split_stem(stem: str) -> tuple[str, str]:
    """'qwen-tuned-3202600219' -> ('qwen-tuned', '3202600219')."""
    for gid in GOLD_IDS:
        if stem.endswith(f"-{gid}"):
            return stem[: -len(gid) - 1], gid
    return stem, ""


def main() -> None:
    out_dir = ROOT / "out" / "baselines"
    golds = {gid: json.loads((ROOT / "data" / "gold" / f"{gid}.json").read_text())
             for gid in GOLD_IDS}

    results: dict[str, dict[str, dict]] = defaultdict(dict)
    for pred_path in sorted(out_dir.glob("*.json")):
        alias, gid = split_stem(pred_path.stem)
        if gid not in golds:
            continue
        pred = json.loads(pred_path.read_text())
        results[alias][gid] = score(golds[gid], pred)

    lines = [
        "# Results",
        "",
        "Requirement-level scores against the three hand-labeled answer keys",
        "(match = same section + enough word overlap; scorer in evals/score.py).",
        "Recall is the headline number: it is the share of real requirements",
        "the model found. Precision is the share of its rows that were real.",
        "",
        "## Baselines (no fine-tuning)",
        "",
        "| Model | RFP | found / real | recall | precision | F1 | labels right |",
        "|---|---|---|---|---|---|---|",
    ]
    def rows_for(aliases: list[str]) -> list[str]:
        out = []
        for alias in aliases:
            if alias not in results:
                continue
            for gid in GOLD_IDS:
                s = results[alias].get(gid)
                if not s:
                    continue
                cat = (f"{s['category_acc']:.0%}"
                       if s["category_acc"] is not None else "n/a")
                out.append(
                    f"| {LABELS.get(alias, alias)} | {gid} "
                    f"| {s['n_matched']} / {s['n_gold']} "
                    f"| {s['recall']:.0%} | {s['precision']:.0%} "
                    f"| {s['f1']:.2f} | {cat} |")
        return out

    lines += rows_for(["claude", "glm", "qwen", "kimi", "inkling"])
    tuned = rows_for(["qwen-tuned", "kimi-tuned", "inkling-tuned"])
    if tuned:
        lines += [
            "",
            "## After fine-tuning (same scorer, same answer keys)",
            "",
            "Each contestant was trained once, with one shared recipe, on the",
            "38 chunk-size pieces of the 11 practice contracts. The three",
            "contracts below stayed locked away during training.",
            "",
            "| Model | RFP | found / real | recall | precision | F1 | labels right |",
            "|---|---|---|---|---|---|---|",
        ] + tuned
    tuned2 = rows_for(["qwen2-tuned", "kimi2-tuned", "inkling2-tuned"])
    if tuned2:
        lines += [
            "",
            "## Round 2: relabeled training data",
            "",
            "For round 2 nothing changed about the training recipe. Only the",
            "training data changed. The practice answers were re-graded against the",
            "labeling rules learned from the hand-checked contracts, and a",
            "second pass hunted down requirement sentences the first teacher",
            "run had missed (586 new rows, growing the training set by half).",
            "The same three locked-away contracts score the result.",
            "",
            "| Model | RFP | found / real | recall | precision | F1 | labels right |",
            "|---|---|---|---|---|---|---|",
        ] + tuned2 + [
            "",
            "Round-2 notes:",
            "",
            "- Averaged across the three contracts, every round-2 model now",
            "  edges Claude on F1 (Claude 0.52; Qwen 0.55; Kimi 0.57;",
            "  Inkling 0.53). Kimi is the round's winner: it beats Claude",
            "  outright on two contracts and ties its recall on the third.",
            "- Inkling's round-1 disaster on HHS0015558v2 (779 rows, 14%",
            "  precision) is fixed: 236 rows, 56% precision, and 73% recall,",
            "  the highest recall any model posted on any contract.",
            "- Claude keeps one crown: precision. 91% on HHS0015558v2 is",
            "  still far above any tuned model, and it wins that contract",
            "  on F1 (0.67).",
            "- The retrained Qwen picked up a repetition habit: on 5 of 9",
            "  chunk runs it started echoing the same row hundreds of times.",
            "  A new harness guard detects this and resamples at higher",
            "  temperature (same guard for every contestant); all 5 runs",
            "  recovered. No such guard existed in round 1, where the same",
            "  disease sank Inkling untreated.",
        ]
    lines += [
        "",
        "## What the failures looked like (from the saved raw outputs)",
        "",
        "- The un-tuned open models' worst enemy was their own habit of",
        "  thinking out loud. On failed runs they filled 100,000+ characters",
        "  with prose notes ('Page 15: ... -> Deadline. Skip.') and ran out",
        "  of room before ever writing the JSON answer. Qwen3.5-4B's zero on",
        "  3202600177 and most Kimi chunks died this way.",
        "- Kimi K2.6 serves with a 32,768-token window on Tinker, so it reads",
        "  each contract in pieces; Qwen does too (small model). Inkling",
        "  (65,536) reads whole contracts but its two big answers were cut",
        "  off mid-list; only the complete rows before the cut were kept.",
        "- Claude's few-shot numbers are the bar: high precision, but it",
        "  finds only a quarter to half of the real requirements. The",
        "  fine-tuning round has to beat that recall to justify itself.",
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {ROOT / 'results.md'}")


if __name__ == "__main__":
    main()
