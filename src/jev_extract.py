"""Compliance-matrix extraction with Jev (TypeSafe System One).

Jev answers questions, it does not write text. So extraction becomes selection:
code splits the RFP into sentences, Jev decides for EVERY sentence whether it
is a requirement and how to label it, and the matrix row's verbatim_text is the
source sentence itself. A row can be wrong, but it can never be invented.

Usage:
  python src/jev_extract.py <rfp_id> [--dry-run] [--limit N]

Raw per-sentence answers (probabilities, latency, tokens) go to
out/jev/<rfp_id>.sentences.json; scripts/jev_eval.py turns them into matrices
at any threshold and scores them against gold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from dotenv import load_dotenv  # noqa: E402

from label_aid import CATEGORY_ENUM, TYPE_ENUM, find_due_date, load_blocks  # noqa: E402

OUT_DIR = ROOT / "out" / "jev"
CONCURRENCY = 12  # well under the 1,200 req/min account limit at ~0.2s/call

# The requirement tests are the ones every bake-off contestant got in
# src/prompts.py EXTRACTION_SYSTEM, restated for a single sentence.
IS_REQUIREMENT = (
    "The sentence in `sentence` belongs in the vendor's compliance matrix for this "
    "government RFP. It does if ANY of these hold: binding language on the vendor "
    "(shall, must, will as a vendor duty, is required to, may not), including passive "
    "or agency-voiced rules that constrain the vendor ('Late submissions will not be "
    "accepted'); advisory language directed at the vendor (should, is encouraged to); "
    "or an administrative fact a bidder must act on (submission or question deadline, "
    "conference date, required form, page limit, copy count, delivery address, "
    "eligibility or certification prerequisite, evaluation criterion or weight). "
    "Judge only `sentence`; `previous` is context. Headings, definitions, program "
    "background and statements of what the agency will do for its own purposes do not "
    "belong."
)

TYPE_CRITERIA = {
    "shall": "Binding obligation on the vendor: shall, must, will (as a vendor duty), is "
             "required to, may not, is prohibited. Passive phrasing counts when it "
             "constrains the vendor.",
    "should": "Advisory to the vendor: should, is encouraged to, is preferred.",
    "info": "No vendor obligation in the sentence's own text: background, definitions, "
            "what the agency will do.",
}

CATEGORY_CRITERIA = {
    "deadline": "Bound to a calendar date or clock time: proposal due, questions due, "
                "conference date, contract start.",
    "eligibility": "A pass/fail qualification the vendor must ALREADY hold to be "
                   "considered: licenses, certifications, registrations, years of "
                   "experience, good standing, not debarred. What you must BE before award.",
    "evaluation": "How proposals are scored or the award decided: criteria, weights, "
                  "points, best value, evaluation committee mechanics.",
    "form": "ONLY the mechanics of the response package: completing, signing or "
            "returning a NAMED form or attachment; page limits, copy counts, formatting, "
            "how the proposal is delivered. A duty to perform work is never form.",
    "deliverable": "Everything else the vendor must do or provide under the contract: "
                   "services, reports, staffing, insurance to maintain, compliance duties. "
                   "The default when nothing above clearly fits.",
}

DISQUALIFIER = (
    "Failing `sentence` would get the bid rejected outright as non-responsive or "
    "ineligible, rather than just lose points or create a contract duty later: a "
    "mandatory form, signature, deadline, minimum qualification, or an explicit "
    "'will be disqualified / rejected / deemed non-responsive'."
)

assert list(TYPE_CRITERIA) == TYPE_ENUM, "TYPE_CRITERIA drifted from schema.json"
assert sorted(CATEGORY_CRITERIA) == sorted(CATEGORY_ENUM), "CATEGORY_CRITERIA drifted from schema.json"


def candidates(txt: str) -> list[dict]:
    """Every splitter sentence, deduped the way the gold review sheets were."""
    out, seen, prev = [], set(), ""
    for page, section, sent in load_blocks(txt, strict_sections=True):
        key = sent.lower()[:120]
        if key not in seen:
            seen.add(key)
            out.append({"idx": len(out), "page": page, "section_ref": section,
                        "sentence": sent, "previous": prev})
        prev = sent
    return out


def build_state(c: dict, rfp_id: str) -> dict:
    return {"document": f"Texas state RFP {rfp_id}", "section": c["section_ref"],
            "page": c["page"], "previous": c["previous"], "sentence": c["sentence"]}


def build_questions():
    from typesafe_sdk import Choice, Noul, NoulCriteria

    return {
        "is_requirement": Noul(instructions=IS_REQUIREMENT),
        "type": Choice(criteria=TYPE_CRITERIA,
                       instructions="How binding is `sentence` on the vendor?"),
        "category": Choice(criteria=CATEGORY_CRITERIA,
                           instructions="What is `sentence` about? Prefer the earliest "
                                        "listed category that fits."),
        "disqualifier": Noul(instructions=DISQUALIFIER,
                             criteria=NoulCriteria(true="pass/fail gate on the bid",
                                                   false="scored, contractual, or not a requirement")),
    }


def to_requirements(sentences: list[dict], threshold: float = 0.5) -> list[dict]:
    """Matrix rows in schema.json form for sentences Jev kept at `threshold`."""
    rows = []
    for s in sentences:
        if s["answers"]["is_requirement"] < threshold:
            continue
        rows.append({
            "id": f"R-{len(rows) + 1:03d}",
            "verbatim_text": s["sentence"],
            "section_ref": s["section_ref"],
            "type": s["answers"]["type"]["choice"],
            "category": s["answers"]["category"]["choice"],
            "due_date": find_due_date(s["sentence"]) or None,
            "page": s["page"],
        })
    return rows


def _flatten(resp) -> dict:
    a = resp.answers
    return {
        "is_requirement": a["is_requirement"].noul,
        "disqualifier": a["disqualifier"].noul,
        "type": {"choice": a["type"].choice, "confidence": a["type"].confidence,
                 "probabilities": dict(a["type"].probabilities)},
        "category": {"choice": a["category"].choice, "confidence": a["category"].confidence,
                     "probabilities": dict(a["category"].probabilities)},
    }


async def run(rfp_id: str, cands: list[dict], model: str | None, on_done=None) -> dict:
    """`on_done()`, if given, is called after each sentence comes back (progress)."""
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    questions = build_questions()
    sem = asyncio.Semaphore(CONCURRENCY)
    client = AsyncTypeSafeClient(model=model, retry=RetryPolicy(max_retries=4))

    async def one(c: dict) -> dict:
        async with sem:
            t0 = time.perf_counter()
            resp = await client.system_one(build_state(c, rfp_id), questions)
            ms = (time.perf_counter() - t0) * 1000
        if on_done:
            on_done()
        return {**c, "answers": _flatten(resp), "latency_ms": round(ms, 1),
                "input_tokens": resp.usage.input_tokens if resp.usage else None,
                "model": resp.model}

    t0 = time.perf_counter()
    try:
        results = await asyncio.gather(*(one(c) for c in cands))
    finally:
        await client.aclose()
    return {"rfp_id": rfp_id, "model": results[0]["model"] if results else model,
            "wall_seconds": round(time.perf_counter() - t0, 2), "sentences": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rfp_id")
    ap.add_argument("--dry-run", action="store_true", help="print one request, call nothing")
    ap.add_argument("--limit", type=int, help="first N sentences only (smoke test)")
    ap.add_argument("--model", default=None, help="pin a version, e.g. jev-1.13.0")
    args = ap.parse_args()

    cands = candidates((ROOT / "data" / "rfps" / f"{args.rfp_id}.txt").read_text())
    if args.limit:
        cands = cands[: args.limit]
    if args.dry_run:
        q = {k: v.model_dump(exclude_none=True) if hasattr(v, "model_dump") else repr(v)
             for k, v in build_questions().items()}
        print(json.dumps({"n_sentences": len(cands), "state": build_state(cands[0], args.rfp_id),
                          "questions": q}, indent=2, default=str))
        return

    load_dotenv(ROOT / ".env")
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY missing: add it to .env")
    model = args.model or os.environ.get("TYPESAFE_DEFAULT_MODEL")
    result = asyncio.run(run(args.rfp_id, cands, model))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f".limit{args.limit}" if args.limit else ""
    out = OUT_DIR / f"{args.rfp_id}{suffix}.sentences.json"
    out.write_text(json.dumps(result, indent=2))
    lat = sorted(s["latency_ms"] for s in result["sentences"])
    toks = sum(s["input_tokens"] or 0 for s in result["sentences"])
    print(f"{args.rfp_id}: {len(lat)} sentences in {result['wall_seconds']}s, "
          f"p50 {lat[len(lat) // 2]:.0f}ms, {toks} input tokens "
          f"(${toks * 0.042 / 1e6:.4f}), model {result['model']} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
