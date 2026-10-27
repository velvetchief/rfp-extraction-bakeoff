"""Deterministic scorer: predicted compliance matrix vs gold.

Requirement-level precision/recall. A predicted requirement matches a gold one
when section_ref matches (normalized) AND verbatim_text overlaps enough
(token-level F1 >= MATCH_THRESHOLD). Each gold row can be claimed once.

Usage: python evals/score.py gold.json predicted.json
"""

import json
import re
import sys

MATCH_THRESHOLD = 0.6


def norm_section(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(section|sec|article|part|attachment|exhibit|appendix)\b", "", s)
    return re.sub(r"[^0-9a-z.]", "", s).strip(".")


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    common = {}
    for t in ta:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in tb:
        if common.get(t, 0) > 0:
            common[t] -= 1
            overlap += 1
    p, r = overlap / len(tb), overlap / len(ta)
    return 0.0 if overlap == 0 else 2 * p * r / (p + r)


def score(gold: dict, pred: dict) -> dict:
    gold_rows = gold["requirements"]
    pred_rows = pred.get("requirements", [])
    claimed = set()
    matched = []
    for pr in pred_rows:
        best, best_f1 = None, 0.0
        for i, gr in enumerate(gold_rows):
            if i in claimed:
                continue
            if norm_section(pr.get("section_ref")) != norm_section(gr["section_ref"]):
                continue
            f1 = token_f1(pr.get("verbatim_text"), gr["verbatim_text"])
            if f1 >= MATCH_THRESHOLD and f1 > best_f1:
                best, best_f1 = i, f1
        if best is not None:
            claimed.add(best)
            gr = gold_rows[best]
            matched.append({
                "gold_id": gr["id"],
                "pred_id": pr.get("id"),
                "text_f1": round(best_f1, 3),
                "type_ok": pr.get("type") == gr["type"],
                "category_ok": pr.get("category") == gr["category"],
                "due_date_ok": pr.get("due_date") == gr.get("due_date"),
            })
    n_gold, n_pred, n_match = len(gold_rows), len(pred_rows), len(matched)
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rfp_id": gold["rfp_id"],
        "n_gold": n_gold,
        "n_pred": n_pred,
        "n_matched": n_match,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "type_acc": round(sum(m["type_ok"] for m in matched) / n_match, 3) if n_match else None,
        "category_acc": round(sum(m["category_ok"] for m in matched) / n_match, 3) if n_match else None,
        "due_date_acc": round(sum(m["due_date_ok"] for m in matched) / n_match, 3) if n_match else None,
        "missed_gold_ids": [g["id"] for i, g in enumerate(gold_rows) if i not in claimed],
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        gold = json.load(f)
    with open(sys.argv[2]) as f:
        pred = json.load(f)
    print(json.dumps(score(gold, pred), indent=2))
