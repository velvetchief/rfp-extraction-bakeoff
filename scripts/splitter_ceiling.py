"""Oracle ceiling for sentence-selection extraction.

Jev can only pick sentences the splitter produces, so the best possible recall
is the share of gold rows that some load_blocks sentence matches under the
same rule evals/score.py uses (same normalized section, token F1 >= 0.6).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "evals")]

from label_aid import load_blocks  # noqa: E402
from score import MATCH_THRESHOLD, norm_section, token_f1  # noqa: E402

RFP_IDS = ["3202600177", "3202600219", "HHS0015558v2"]


def ceiling(rfp_id: str, strict: bool) -> dict:
    blocks = load_blocks((ROOT / "data" / "rfps" / f"{rfp_id}.txt").read_text(), strict_sections=strict)
    gold = json.loads((ROOT / "data" / "gold" / f"{rfp_id}.json").read_text())["requirements"]
    reach, reach_any_section, misses = 0, 0, []
    for g in gold:
        sec = norm_section(g["section_ref"])
        best_same = max((token_f1(g["verbatim_text"], s) for _, ref, s in blocks
                         if norm_section(ref) == sec), default=0.0)
        best_any = max(token_f1(g["verbatim_text"], s) for _, _, s in blocks)
        reach += best_same >= MATCH_THRESHOLD
        reach_any_section += best_any >= MATCH_THRESHOLD
        if best_same < MATCH_THRESHOLD:
            misses.append({"id": g["id"], "section": g["section_ref"],
                           "best_same": round(best_same, 2), "best_any": round(best_any, 2),
                           "text": g["verbatim_text"][:120]})
    return {"rfp": rfp_id, "sentences": len(blocks), "gold": len(gold),
            "reachable": reach, "reachable_ignoring_section": reach_any_section,
            "misses": misses}


if __name__ == "__main__":
    for strict in (False, True):
        print("strict_sections =", strict)
        rows = [ceiling(r, strict) for r in RFP_IDS]
        tot = sum(r["gold"] for r in rows)
        print(f"{'ALL':>14}  reachable={sum(r['reachable'] for r in rows)}/{tot} "
              f"({sum(r['reachable'] for r in rows)/tot:.0%})")
        for r in rows:
            print(f"{r['rfp']:>14}  sentences={r['sentences']:4}  gold={r['gold']:3}  "
                  f"reachable={r['reachable']:3} ({r['reachable']/r['gold']:.0%})  "
                  f"ignoring-section={r['reachable_ignoring_section']/r['gold']:.0%}")
    out = ROOT / "out" / "jev" / "splitter_ceiling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"misses -> {out.relative_to(ROOT)}")
