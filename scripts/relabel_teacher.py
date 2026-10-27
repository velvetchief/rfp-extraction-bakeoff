"""Round 2, step 1: re-label the teacher matrices to the gold labeling convention.

Round-1 diagnosis: the teacher (Claude) used a different labeling convention
than the gold answer keys — gold is 68% deliverable / 15% form, the teacher
was 45% / 29%, and the tuned models faithfully learned the teacher's
convention, which then scored as label errors. This script keeps every row's
verbatim_text / section_ref / page untouched and asks Claude to re-assign
ONLY type, category, and due_date under an explicit rubric.

The rubric below is written in plain language with synthetic examples. It was
derived by reading the gold keys, but contains NO text from any gold or
held-out RFP. Nothing here can leak the held-out answers into training data.

Usage (needs ANTHROPIC_API_KEY in .env):
  .venv/bin/python scripts/relabel_teacher.py            # all of data/train/
  .venv/bin/python scripts/relabel_teacher.py HHS0011234 # just one
  .venv/bin/python scripts/relabel_teacher.py --force    # redo existing

Writes: data/train2/<rfp_id>.json  (same schema, relabeled rows)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baseline import load_env  # noqa: E402
from teacher import strict  # noqa: E402

TRAIN_DIR = ROOT / "data" / "train"
OUT_DIR = ROOT / "data" / "train2"
HELD_OUT = {"3202600219", "HHS0015558v2", "3202600177"}

MODEL = "claude-opus-4-8"
BATCH = 40  # rows per API call

# The labeling convention, shared with scripts/mine_misses.py. Derived by
# reading the gold keys but written fresh with synthetic examples — no gold
# text appears here.
LABEL_RUBRIC = """\
TYPE — how binding the row is:
- "shall": a binding obligation on the vendor — shall, must, will (as a vendor \
duty), is required to, may not, is prohibited. Passive phrasing still counts \
when it constrains the vendor ("Late submissions will not be accepted" binds \
the vendor to submit on time).
- "should": advisory — should, is encouraged to, is preferred.
- "info": a fact with no vendor obligation in the row's own text — program \
background, definitions, statements about what the agency will do. If the row \
itself doesn't constrain the vendor, it's info even when neighboring rows do.

CATEGORY — what the row is about. Apply this decision order:
1. "deadline": the obligation is bound to a calendar date or clock time \
(proposal due, questions due, conference date, contract start). Carries \
due_date when a specific date appears in the row's text.
2. "eligibility": a pass/fail minimum qualification the vendor must ALREADY \
hold to be considered — licenses, certifications, registrations, years of \
experience, good standing, not being debarred. What you must BE before award, \
not what you must do after it.
3. "evaluation": how proposals will be scored or the award decided — criteria, \
weights, points, best-value language, evaluation committee mechanics.
4. "form": ONLY the mechanics of the response package itself — completing, \
signing, or returning a NAMED form or attachment; page limits, copy counts, \
formatting, delivery method for the proposal. A duty to perform work is never \
"form" just because a document is involved in the work.
5. "deliverable": everything else the vendor must do or provide under the \
contract — services, reports, staffing, insurance to maintain, compliance \
duties, work products. This is the DEFAULT bucket for obligations: when a row \
doesn't clearly fit 1-4, it is a deliverable.

The three specific mistakes you are correcting (from the draft labels):
- Work duties that mention a document get mislabeled "form". "The Contractor \
shall deliver a monthly status report to the program manager" is a \
deliverable. "The Respondent shall complete and return Attachment A, Bid \
Certification" is form.
- Contract-performance conditions get mislabeled "eligibility". "The \
Contractor shall maintain commercial general liability insurance of $1M \
throughout the term" is a deliverable — it is performed during the contract. \
"The Respondent must hold a current Texas electrician license" is eligibility.
- Binding rows get mislabeled "info" because the sentence is passive or \
agency-voiced. "Proposals received after the deadline will be rejected" is \
type shall (category deadline). Reserve "info" for rows with no obligation \
at all, like "The Department serves approximately 40,000 clients annually" \
(type info, category deliverable is wrong there — see rule below).

Category for "info" rows: still pick the closest bucket by topic — a fact \
about the evaluation process is evaluation, a stated conference date with no \
obligation is deadline, program background with no better fit is deliverable.

due_date: ISO YYYY-MM-DD only when that specific date appears in the row's \
own text; otherwise null. Never infer a date from outside the row.\
"""

RELABEL_SYSTEM = """\
You re-label rows of a draft compliance matrix extracted from a US state/local \
government RFP.

Each row is a candidate requirement with its exact text and section reference. \
Your ONLY job is to assign the correct `type` and `category` to each row under \
the convention below, plus an ISO `due_date` when the row's own text carries a \
specific calendar date. Do not rewrite text, do not drop rows, do not add rows. \
Return every row id you were given, in the same order.

""" + LABEL_RUBRIC

RELABEL_USER = """\
Solicitation ID: {rfp_id}

Re-label these draft rows. Return a JSON object {{"rows": [...]}} with one \
entry per input row — same ids, same order — each carrying your corrected \
type, category, and due_date.

ROWS:
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
                "required": ["id", "type", "category", "due_date"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"enum": ["shall", "should", "info"]},
                    "category": {"enum": ["eligibility", "deliverable",
                                          "deadline", "form", "evaluation"]},
                    "due_date": {"type": ["string", "null"]},
                },
            },
        }
    },
}


def relabel_batch(client, rfp_id: str, rows: list[dict]) -> dict[str, dict]:
    """One API call; returns {row_id: {type, category, due_date}}."""
    payload = [{"id": r["id"], "section_ref": r["section_ref"],
                "verbatim_text": r["verbatim_text"]} for r in rows]
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=RELABEL_SYSTEM,
        output_config={"format": {"type": "json_schema",
                                  "schema": strict(json.loads(json.dumps(RESPONSE_SCHEMA)))}},
        messages=[{
            "role": "user",
            "content": RELABEL_USER.format(
                rfp_id=rfp_id,
                rows_json=json.dumps(payload, ensure_ascii=False, indent=1)),
        }],
    ) as stream:
        message = stream.get_final_message()
    text = "".join(b.text for b in message.content if b.type == "text")
    out = {r["id"]: r for r in json.loads(text)["rows"]}
    want, got = {r["id"] for r in rows}, set(out)
    if want != got:
        raise RuntimeError(f"{rfp_id}: relabel ids mismatch — "
                           f"missing {sorted(want - got)}, extra {sorted(got - want)}")
    return out


def relabel_rfp(client, matrix_path: Path) -> tuple[int, int, int]:
    rfp_id = matrix_path.stem
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["requirements"]
    changed_t = changed_c = changed_d = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        labels = relabel_batch(client, rfp_id, batch)
        for r in batch:
            new = labels[r["id"]]
            if new["type"] != r["type"]:
                changed_t += 1
            if new["category"] != r["category"]:
                changed_c += 1
            if new["due_date"] != r.get("due_date"):
                changed_d += 1
            r["type"] = new["type"]
            r["category"] = new["category"]
            r["due_date"] = new["due_date"]
        print(f"  {rfp_id}: {min(i + BATCH, len(rows))}/{len(rows)} rows")
    out_path = OUT_DIR / f"{rfp_id}.json"
    out_path.write_text(json.dumps(matrix, indent=2))
    return changed_t, changed_c, changed_d


def main() -> None:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("needs ANTHROPIC_API_KEY in .env")
    from anthropic import Anthropic
    client = Anthropic()

    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    paths = ([TRAIN_DIR / f"{a}.json" for a in args] if args
             else sorted(TRAIN_DIR.glob("*.json")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tot_rows = tot_t = tot_c = tot_d = 0
    for p in paths:
        if not p.exists():
            sys.exit(f"no teacher matrix at {p}")
        assert p.stem not in HELD_OUT, f"{p.stem} must stay held out"
        if (OUT_DIR / p.name).exists() and not force:
            print(f"{p.stem}: already in data/train2/ (use --force to redo)")
            continue
        n = len(json.loads(p.read_text())["requirements"])
        t, c, d = relabel_rfp(client, p)
        tot_rows += n
        tot_t += t
        tot_c += c
        tot_d += d
        print(f"{p.stem}: {n} rows, changed type={t} category={c} due_date={d}")
    if tot_rows:
        print(f"\ntotal: {tot_rows} rows — type {tot_t} ({tot_t / tot_rows:.0%}), "
              f"category {tot_c} ({tot_c / tot_rows:.0%}), "
              f"due_date {tot_d} -> data/train2/")


if __name__ == "__main__":
    main()
