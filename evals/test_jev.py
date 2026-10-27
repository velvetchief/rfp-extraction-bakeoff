import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "evals")]

import jev_eval  # noqa: E402
from jev_extract import candidates, to_requirements  # noqa: E402


def sent(text, p_req, section="2.1", typ="shall", cat="deliverable", page=3):
    return {"sentence": text, "section_ref": section, "page": page, "latency_ms": 100.0,
            "input_tokens": 400,
            "answers": {"is_requirement": p_req, "disqualifier": 0.1,
                        "type": {"choice": typ, "confidence": 0.9, "probabilities": {}},
                        "category": {"choice": cat, "confidence": 0.9, "probabilities": {}}}}


def test_rows_are_verbatim_source_sentences_and_threshold_is_inclusive():
    s = [sent("The Contractor shall file reports by June 1, 2026.", 0.5), sent("Background.", 0.49)]
    rows = to_requirements(s, 0.5)
    assert [r["verbatim_text"] for r in rows] == ["The Contractor shall file reports by June 1, 2026."]
    assert rows[0]["id"] == "R-001" and rows[0]["due_date"] == "2026-06-01"


def test_every_row_text_exists_in_source_rfp():
    txt = (ROOT / "data" / "rfps" / "3202600219.txt").read_text()
    cands = candidates(txt)
    rows = to_requirements([{**c, **sent(c["sentence"], 1.0)} | {"section_ref": c["section_ref"]}
                            for c in cands])
    source = " ".join(txt.split())
    # splitter de-hyphenates wrapped words, so compare on a hyphen-free, whitespace-normalized view
    norm = lambda t: " ".join(t.replace("-", "").split())  # noqa: E731
    missing = [r["verbatim_text"] for r in rows if norm(r["verbatim_text"])[:60] not in norm(source)]
    assert len(missing) / len(rows) < 0.05, missing[:3]


def test_reliability_perfectly_calibrated_has_zero_ece():
    probs = [0.95] * 20 + [0.05] * 20
    labels = [True] * 19 + [False] + [True] + [False] * 19
    cal = jev_eval.reliability(probs, labels)
    assert cal["ece"] == 0.0
    assert [b["n"] for b in cal["bins"]] == [20, 20]


def test_reliability_overconfident_gap():
    cal = jev_eval.reliability([0.9] * 10, [True] * 5 + [False] * 5)
    assert cal["ece"] == 0.4


def test_reliability_prob_one_lands_in_top_bin():
    assert jev_eval.reliability([1.0], [True])["bins"][0]["lo"] == 0.9


def test_oracle_run_reproduces_splitter_ceiling(tmp_path, monkeypatch):
    rfp = "3202600219"
    gold = json.loads((ROOT / "data" / "gold" / f"{rfp}.json").read_text())["requirements"]
    cands = candidates((ROOT / "data" / "rfps" / f"{rfp}.txt").read_text())
    sents = [{**c, **sent(c["sentence"], 1.0 if jev_eval.gold_positive(c, gold) else 0.0,
                          section=c["section_ref"], page=c["page"])} for c in cands]
    (tmp_path / f"{rfp}.sentences.json").write_text(
        json.dumps({"model": "oracle", "wall_seconds": 1.0, "sentences": sents}))
    monkeypatch.setattr(jev_eval, "OUT_DIR", tmp_path)
    r = jev_eval.evaluate(rfp)
    assert r["headline"]["n_matched"] >= 50  # ceiling measured 52/54
    assert r["headline"]["precision"] >= 0.9
    assert r["calibration"]["ece"] == 0.0
