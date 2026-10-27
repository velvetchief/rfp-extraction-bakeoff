"""Score Jev extraction against gold: F1, threshold sweep, calibration.

  python scripts/jev_eval.py            # all three gold RFPs
  python scripts/jev_eval.py --limit 40 # score smoke-test outputs

Reads out/jev/<rfp>.sentences.json from src/jev_extract.py and writes
out/jev/eval.json plus a markdown table to stdout.

Calibration asks: of the sentences Jev called a requirement with ~0.9, were
~90% actually in the gold matrix? A sentence counts as a real requirement when
it would match some gold row under the scorer's own rule (same normalized
section, token F1 >= 0.6).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evals"), str(ROOT / "scripts")]

from jev_extract import OUT_DIR, to_requirements  # noqa: E402
from score import MATCH_THRESHOLD, norm_section, score, token_f1  # noqa: E402

RFP_IDS = ["3202600177", "3202600219", "HHS0015558v2"]
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
BASELINE_MACRO_F1 = {"Claude few-shot": 0.52, "inkling2": 0.53, "qwen2": 0.55, "kimi2-tuned": 0.57}
FALSIFIER_F1 = 0.48  # below this, Jev lost the rematch
FALSIFIER_CAL_GAP = 0.10  # top bin off by more than this, "calibrated" is not earned here


def gold_positive(sentence: dict, gold_rows: list[dict]) -> bool:
    sec = norm_section(sentence["section_ref"])
    return any(norm_section(g["section_ref"]) == sec
               and token_f1(sentence["sentence"], g["verbatim_text"]) >= MATCH_THRESHOLD
               for g in gold_rows)


def reliability(probs: list[float], labels: list[bool], n_bins: int = 10) -> dict:
    """Equal-width bins over [0, 1]. Returns per-bin stats plus expected
    calibration error (bin-size-weighted mean of |mean prob - hit rate|)."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            continue
        mean_p = sum(probs[i] for i in idx) / len(idx)
        hit = sum(labels[i] for i in idx) / len(idx)
        bins.append({"lo": lo, "hi": hi, "n": len(idx), "mean_prob": round(mean_p, 3),
                     "hit_rate": round(hit, 3), "gap": round(abs(mean_p - hit), 3)})
    n = len(probs)
    ece = sum(b["n"] / n * b["gap"] for b in bins) if n else 0.0
    return {"bins": bins, "ece": round(ece, 3)}


def evaluate(rfp_id: str, suffix: str = "") -> dict:
    run = json.loads((OUT_DIR / f"{rfp_id}{suffix}.sentences.json").read_text())
    gold = json.loads((ROOT / "data" / "gold" / f"{rfp_id}.json").read_text())
    sents = run["sentences"]
    headline = score(gold, {"requirements": to_requirements(sents, 0.5)})
    sweep = [{"threshold": t, **{k: score(gold, {"requirements": to_requirements(sents, t)})[k]
                                 for k in ("n_pred", "precision", "recall", "f1")}}
             for t in THRESHOLDS]
    labels = [gold_positive(s, gold["requirements"]) for s in sents]
    lat = sorted(s["latency_ms"] for s in sents)
    toks = sum(s["input_tokens"] or 0 for s in sents)
    return {
        "rfp_id": rfp_id, "model": run["model"], "n_sentences": len(sents),
        "headline": {k: v for k, v in headline.items() if k != "missed_gold_ids"},
        "missed_gold_ids": headline["missed_gold_ids"],
        "sweep": sweep,
        "calibration": reliability([s["answers"]["is_requirement"] for s in sents], labels),
        "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95)]},
        "wall_seconds": run["wall_seconds"],
        "input_tokens": toks, "cost_usd": round(toks * 0.042 / 1e6, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    suffix = f".limit{args.limit}" if args.limit else ""
    runs = [evaluate(r, suffix) for r in RFP_IDS if (OUT_DIR / f"{r}{suffix}.sentences.json").exists()]
    if not runs:
        sys.exit("no Jev outputs yet: run src/jev_extract.py first")

    macro = sum(r["headline"]["f1"] for r in runs) / len(runs)
    top = [b for r in runs for b in r["calibration"]["bins"] if b["lo"] >= 0.9]
    top_gap = (abs(sum(b["mean_prob"] * b["n"] for b in top) - sum(b["hit_rate"] * b["n"] for b in top))
               / sum(b["n"] for b in top)) if top else None
    verdict = {
        "macro_f1": round(macro, 3),
        "top_bin_gap": round(top_gap, 3) if top_gap is not None else None,
        "f1_holds": macro >= FALSIFIER_F1,
        "calibration_holds": top_gap is not None and top_gap <= FALSIFIER_CAL_GAP,
        "complete": len(runs) == len(RFP_IDS) and not args.limit,
    }
    (OUT_DIR / f"eval{suffix}.json").write_text(json.dumps({"verdict": verdict, "runs": runs}, indent=2))

    print("| RFP | matched / gold | precision | recall | F1 | type acc | category acc "
          "| ECE | p50 ms | cost |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        h = r["headline"]
        print(f"| {r['rfp_id']} | {h['n_matched']} / {h['n_gold']} "
              f"| {h['precision']:.0%} | {h['recall']:.0%} "
              f"| {h['f1']:.2f} | {h['type_acc'] or 0:.0%} | {h['category_acc'] or 0:.0%} "
              f"| {r['calibration']['ece']:.3f} | {r['latency_ms']['p50']:.0f} | ${r['cost_usd']:.4f} |")
    print(f"\nJev macro-F1 {macro:.2f} vs " + ", ".join(f"{k} {v:.2f}" for k, v in BASELINE_MACRO_F1.items()))
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
