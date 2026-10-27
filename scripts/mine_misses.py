"""Round 2, step 2: mine the requirements the teacher missed and add them.

Round-1 diagnosis: the tuned models' recall ceiling (~43-51%) is inherited —
the teacher matrices carry Claude's blind spots (modal-free requirements,
truncated sections, glued headings), so the students can't learn what the
teacher never showed them. This script runs label_aid's recall-first sentence
extractor over each TRAINING RFP, keeps the candidates that don't match any
existing teacher row, has Claude adjudicate keep/drop and label the keepers
under the same rubric as relabel_teacher, and appends them.

Training RFPs only — held-out text never enters this pipeline.
Run AFTER relabel_teacher.py (reads and rewrites data/train2/).
Naturally idempotent: appended rows match their own candidates on rerun.

Usage (needs ANTHROPIC_API_KEY in .env):
  .venv/bin/python scripts/mine_misses.py            # all of data/train2/
  .venv/bin/python scripts/mine_misses.py HHS0011234 # just one
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline import load_env  # noqa: E402
from label_aid import (DEADLINE_RE, SHALL_RE, SHOULD_RE, find_due_date,  # noqa: E402
                       load_blocks)
from relabel_teacher import LABEL_RUBRIC  # noqa: E402
from score import token_f1  # noqa: E402
from teacher import strict  # noqa: E402

TRAIN2_DIR = ROOT / "data" / "train2"
RFP_DIR = ROOT / "data" / "rfps"
HELD_OUT = {"3202600219", "HHS0015558v2", "3202600177"}

MODEL = "claude-opus-4-8"
BATCH = 30  # candidates per API call
# A candidate counts as already-covered when it overlaps an existing teacher
# row this much. Deliberately below the scorer's 0.6: borderline overlaps go
# to adjudication rather than silently assumed covered.
COVERED_F1 = 0.5

MINE_SYSTEM = """\
You adjudicate candidate requirements for a compliance matrix drafted from a \
US state/local government RFP.

A recall-first extractor pulled every sentence that carries obligation \
language from the RFP; a first drafting pass already captured many of them, \
and the sentences below matched none of the drafted rows. For each candidate, \
decide keep or drop.

KEEP a candidate when it belongs in a compliance matrix — it meets ANY of:
- Binding language on the vendor: shall, must, will (vendor obligation), is \
required to, may not.
- Advisory language directed at the vendor: should, is encouraged to.
- An administrative fact a bidder must act on: submission or question \
deadlines, conference dates, required forms, page limits, copy counts, \
delivery addresses, eligibility prerequisites, evaluation criteria or weights.

DROP a candidate when it is:
- Boilerplate with no vendor action: definitions, agency background, \
severability/governing-law legalese that binds the agency or nobody.
- A table-of-contents line, heading fragment, footer, or garbled extraction.
- A duplicate restatement of another candidate in this batch (keep the \
clearest one).

For every KEPT candidate, assign type, category, and due_date under this \
convention:

""" + LABEL_RUBRIC

MINE_USER = """\
Solicitation ID: {rfp_id}

Adjudicate these candidates. Return {{"rows": [...]}} with one entry per \
candidate — same ids, same order — each with keep (true/false) and, when \
keep is true, your type, category, and due_date. For dropped candidates \
still fill type/category with your best guess (they are ignored).

CANDIDATES:
{rows_json}\
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["rows"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "keep", "type", "category", "due_date"],
                "properties": {
                    "id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "type": {"enum": ["shall", "should", "info"]},
                    "category": {"enum": ["eligibility", "deliverable",
                                          "deadline", "form", "evaluation"]},
                    "due_date": {"type": ["string", "null"]},
                },
            },
        }
    },
}


def candidates_for(rfp_id: str) -> list[dict]:
    """label_aid's recall-first pass: every obligation-bearing sentence."""
    txt = (RFP_DIR / f"{rfp_id}.txt").read_text()
    out, seen = [], set()
    for page, section, sent in load_blocks(txt):
        is_req = bool(SHALL_RE.search(sent) or SHOULD_RE.search(sent))
        is_dateinfo = bool(find_due_date(sent) and DEADLINE_RE.search(sent))
        if not (is_req or is_dateinfo):
            continue
        key = sent.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"C-{len(out) + 1:03d}", "page": page,
                    "section_ref": section, "verbatim_text": sent})
    return out


def uncovered(cands: list[dict], rows: list[dict]) -> list[dict]:
    """Candidates whose text overlaps no existing row at COVERED_F1.

    Text-only matching (no section check) on purpose: the teacher and the
    heading tracker cite sections differently, and a section mismatch must
    not resurrect a sentence the teacher already has.
    """
    texts = [r["verbatim_text"] for r in rows]
    return [c for c in cands
            if not any(token_f1(c["verbatim_text"], t) >= COVERED_F1
                       for t in texts)]


def adjudicate(client, rfp_id: str, cands: list[dict]) -> list[dict]:
    """Claude keeps/drops + labels; returns the kept rows, schema-shaped."""
    kept = []
    for i in range(0, len(cands), BATCH):
        batch = cands[i:i + BATCH]
        payload = [{"id": c["id"], "section_ref": c["section_ref"],
                    "verbatim_text": c["verbatim_text"]} for c in batch]
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=MINE_SYSTEM,
            output_config={"format": {"type": "json_schema",
                                      "schema": strict(json.loads(json.dumps(RESPONSE_SCHEMA)))}},
            messages=[{
                "role": "user",
                "content": MINE_USER.format(
                    rfp_id=rfp_id,
                    rows_json=json.dumps(payload, ensure_ascii=False, indent=1)),
            }],
        ) as stream:
            message = stream.get_final_message()
        text = "".join(b.text for b in message.content if b.type == "text")
        verdicts = {r["id"]: r for r in json.loads(text)["rows"]}
        want, got = {c["id"] for c in batch}, set(verdicts)
        if want != got:
            raise RuntimeError(f"{rfp_id}: adjudication ids mismatch — "
                               f"missing {sorted(want - got)}, extra {sorted(got - want)}")
        for c in batch:
            v = verdicts[c["id"]]
            if not v["keep"]:
                continue
            kept.append({
                "id": "TBD",  # renumbered after merge
                "verbatim_text": c["verbatim_text"],
                "section_ref": c["section_ref"],
                "type": v["type"],
                "category": v["category"],
                "due_date": v["due_date"],
                "page": c["page"],
            })
        print(f"  {rfp_id}: adjudicated {min(i + BATCH, len(cands))}/{len(cands)}")
    return kept


def mine_rfp(client, matrix_path: Path) -> tuple[int, int, int]:
    rfp_id = matrix_path.stem
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["requirements"]
    cands = candidates_for(rfp_id)
    new = uncovered(cands, rows)
    if not new:
        return len(cands), 0, 0
    kept = adjudicate(client, rfp_id, new)
    rows.extend(kept)
    rows.sort(key=lambda r: r["page"])  # stable: keeps in-page order
    for i, r in enumerate(rows, 1):
        r["id"] = f"R-{i:03d}"
    matrix_path.write_text(json.dumps(matrix, indent=2))
    return len(cands), len(new), len(kept)


def main() -> None:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("needs ANTHROPIC_API_KEY in .env")
    from anthropic import Anthropic
    client = Anthropic()

    args = sys.argv[1:]
    paths = ([TRAIN2_DIR / f"{a}.json" for a in args] if args
             else sorted(TRAIN2_DIR.glob("*.json")))
    if not paths:
        sys.exit("data/train2/ is empty — run scripts/relabel_teacher.py first")

    tot_new = tot_kept = 0
    for p in paths:
        if not p.exists():
            sys.exit(f"no relabeled matrix at {p}")
        assert p.stem not in HELD_OUT, f"{p.stem} must stay held out"
        n_before = len(json.loads(p.read_text())["requirements"])
        n_cand, n_new, n_kept = mine_rfp(client, p)
        tot_new += n_new
        tot_kept += n_kept
        print(f"{p.stem}: {n_cand} candidates, {n_new} unmatched, "
              f"{n_kept} kept ({n_before} -> {n_before + n_kept} rows)")
    print(f"\ntotal: {tot_new} unmatched candidates, {tot_kept} added")


if __name__ == "__main__":
    main()
