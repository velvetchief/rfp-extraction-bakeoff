"""Gold-labeling aid: turn a raw RFP into a review sheet you tick and correct.

This does NOT write the answer key for you — it does the tedious part (reading
every page, rejoining PDF-wrapped lines, and pulling every sentence that carries
a "shall / must / required" obligation) and hands you a spreadsheet where each
candidate requirement is one row, pre-filled with a best-guess type/category.
You stay the judge: delete the rows that aren't real requirements, fix any
guessed field, add anything the extractor missed, then convert to gold JSON.

Recall-first by design: when unsure, it INCLUDES the sentence. Over-inclusion is
a 2-second delete; a miss is a hole in your answer key.

Two steps:

  1. Build the review sheet (one CSV per RFP):
       python3 scripts/label_aid.py 3202600219 HHS0015558v2 3202600177
     -> data/gold/<id>.review.csv   (open in Numbers/Excel, edit, save as CSV)

     In each row, column `keep` is pre-set to y. Set it to n (or delete the row)
     to drop a false positive. Edit type/category/due_date/section as needed.
     To ADD a missed requirement, copy a row and rewrite verbatim + fields.

  2. Convert your edited sheet to the gold answer key:
       python3 scripts/label_aid.py --to-gold 3202600219
     -> data/gold/3202600219.json   (renumbered R-001.., schema-validated)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RFP_DIR = ROOT / "data" / "rfps"
GOLD_DIR = ROOT / "data" / "gold"
SCHEMA = json.loads((ROOT / "schema.json").read_text())

CSV_COLS = ["keep", "id", "page", "section", "type", "category", "due_date", "verbatim"]

_REQ_PROPS = SCHEMA["properties"]["requirements"]["items"]["properties"]
TYPE_ENUM = _REQ_PROPS["type"]["enum"]
CATEGORY_ENUM = _REQ_PROPS["category"]["enum"]

# --- line-level cleanup ----------------------------------------------------
PAGE_RE = re.compile(r"^\[page (\d+)\]\s*$")
# running header/footer like "IFB No. 3202600219 Page 2 of 25"
FOOTER_RE = re.compile(r"Page\s+\d+\s+of\s+\d+", re.I)
# table-of-contents / dotted-leader lines: "1.1 Purpose ........... 7"
TOC_RE = re.compile(r"\.{4,}\s*\d+\s*$")
# a section heading: "3.2 Contract Award", "Article III. Scope", "4.1 ..."
HEADING_RE = re.compile(
    r"^\s*(?P<ref>Article\s+[IVXLC]+|Section\s+\d+[\w.]*|\d+(?:\.\d+){0,3})[.)]?\s+[A-Z]"
)

# --- requirement detection (recall-first) ----------------------------------
SHALL_RE = re.compile(
    r"\b(shall not|must not|may not|shall|must|is required to|are required to|"
    r"will be required|responsible for|agrees? to|is prohibited|are prohibited|"
    r"at a minimum|no later than)\b",
    re.I,
)
SHOULD_RE = re.compile(r"\bshould\b", re.I)

FORM_RE = re.compile(
    r"\b(attachment|exhibit|appendix|form|HUB Subcontracting Plan|\bHSP\b|"
    r"sign and return|complete and (submit|return)|certification|certificate|"
    r"notariz|W-9|submit (a|the|its|one|two|three|an original))\b",
    re.I,
)
EVAL_RE = re.compile(
    r"\b(evaluat|scored|scoring|best value|award(ed|ing)?|responsive|"
    r"responsibl(e|ity) determination|selection criteria|weight(ed|ing)?|points?)\b",
    re.I,
)
ELIG_RE = re.compile(
    r"\b(eligib|qualif|minimum qualification|must meet|licensed|registered|"
    r"in good standing|authorized to do business|HUB status|debarr)\b",
    re.I,
)
DEADLINE_RE = re.compile(
    r"\b(due|deadline|no later than|by\s+\d{1,2}:\d{2}|submission (date|deadline)|"
    r"received by|postmark)\b",
    re.I,
)

MONTHS = ("january february march april may june july august september october "
          "november december").split()
MONTH_IDX = {m: i + 1 for i, m in enumerate(MONTHS)}
DATE_WORD_RE = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})\s*,?\s*(\d{4})\b", re.I
)
DATE_NUM_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def find_due_date(text: str) -> str:
    m = DATE_WORD_RE.search(text)
    if m:
        return f"{int(m.group(3)):04d}-{MONTH_IDX[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    m = DATE_NUM_RE.search(text)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return ""


def guess_type(sent: str) -> str:
    if SHALL_RE.search(sent):
        return "shall"
    if SHOULD_RE.search(sent):
        return "should"
    return "info"


def guess_category(sent: str) -> str:
    # order matters: a dated form-submission reads as a deadline first
    if DEADLINE_RE.search(sent) or find_due_date(sent):
        return "deadline"
    if FORM_RE.search(sent):
        return "form"
    if EVAL_RE.search(sent):
        return "evaluation"
    if ELIG_RE.search(sent):
        return "eligibility"
    return "deliverable"


# --- rebuild wrapped lines into pages/sections/sentences -------------------
def _is_list_number(ref: str) -> bool:
    """A bare integer "heading" ("1. Educate ...", "1100 West 49th") is a
    numbered list item or street address. Chapters in these RFPs are headed
    "ARTICLE II" or dotted refs ("2.2.1"), never a lone integer."""
    return ref.isdigit()


def load_blocks(txt: str, strict_sections: bool = False) -> list[tuple[int, str, str]]:
    """Return (page, section_ref, sentence) for every candidate sentence.

    PDF text is hard-wrapped mid-sentence, so we rejoin lines within a page/
    section into paragraphs (de-hyphenating line-break splits), then sentence-
    split, before any requirement filtering.

    strict_sections rejects bare-integer "headings" that are really list items
    or addresses (see _is_list_number). Off by default so the splitter that
    built the gold review sheets stays reproducible.
    """
    page = 1
    section = "Cover"
    # accumulate (page, section, text) paragraphs
    paras: list[tuple[int, str, list[str]]] = []
    buf: list[str] = []
    buf_page = 1
    buf_section = "Cover"

    def flush():
        nonlocal buf
        if buf:
            paras.append((buf_page, buf_section, buf))
            buf = []

    for raw in txt.splitlines():
        line = raw.rstrip()
        pm = PAGE_RE.match(line)
        if pm:
            flush()
            page = int(pm.group(1))
            continue
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if FOOTER_RE.search(stripped) and len(stripped) < 60:
            continue  # running header/footer noise
        if TOC_RE.search(stripped):
            continue  # table-of-contents entry
        hm = HEADING_RE.match(stripped)
        if (hm and len(stripped.split()) <= 12 and not stripped.endswith((".", ";", ":"))
                and not (strict_sections and _is_list_number(hm.group("ref").strip()))):
            flush()
            section = hm.group("ref").strip()
            # keep the heading text as its own block too (some are requirements)
        if not buf:
            buf_page, buf_section = page, section
        buf.append(stripped)

    flush()

    out: list[tuple[int, str, str]] = []
    for p, sec, lines in paras:
        joined = _join_wrapped(lines)
        for sent in _split_sentences(joined):
            sent = sent.strip()
            if len(sent.split()) >= 3:
                out.append((p, sec, sent))
    return out


def _join_wrapped(lines: list[str]) -> str:
    text = ""
    for ln in lines:
        if not text:
            text = ln
        elif text.endswith("-"):
            text = text[:-1] + ln  # de-hyphenate "plug-\nin" -> "plugin"
        else:
            text = text + " " + ln
    return re.sub(r"\s+", " ", text).strip()


_ABBR = re.compile(r"(a\.m|p\.m|No|U\.S|Inc|Ltd|Corp|St|Mr|Mrs|Dr|vs|e\.g|i\.e|"
                   r"Sec|Art|§|\b[A-Z])\.$", re.I)


def _split_sentences(text: str) -> list[str]:
    # split on ., ;, : followed by whitespace + an opening (capital/quote/paren),
    # but not after a known abbreviation or a single initial.
    parts = re.split(r'(?<=[.;:])\s+(?=[A-Z“"(])', text)
    merged: list[str] = []
    for part in parts:
        if merged and _ABBR.search(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def build_sheet(rfp_id: str) -> None:
    txt_path = RFP_DIR / f"{rfp_id}.txt"
    if not txt_path.exists():
        sys.exit(f"no text for {rfp_id} at {txt_path}")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: set[str] = set()
    n = 0
    for page, section, sent in load_blocks(txt_path.read_text()):
        is_req = bool(SHALL_RE.search(sent) or SHOULD_RE.search(sent))
        is_dateinfo = bool(find_due_date(sent) and DEADLINE_RE.search(sent))
        if not (is_req or is_dateinfo):
            continue
        key = sent.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        n += 1
        rows.append({
            "keep": "y",
            "id": f"R-{n:03d}",
            "page": page,
            "section": section,
            "type": guess_type(sent),
            "category": guess_category(sent),
            "due_date": find_due_date(sent),
            "verbatim": sent,
        })
    out_path = GOLD_DIR / f"{rfp_id}.review.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{rfp_id}: {len(rows)} candidate requirements -> {out_path.relative_to(ROOT)}")


def _truthy(v: str) -> bool:
    return v.strip().lower() in {"y", "yes", "1", "true", "x", "keep"}


def to_gold(rfp_id: str) -> None:
    csv_path = GOLD_DIR / f"{rfp_id}.review.csv"
    if not csv_path.exists():
        sys.exit(f"no review sheet at {csv_path} — run: python3 scripts/label_aid.py {rfp_id}")
    reqs = []
    errors = []
    with csv_path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 2):  # row 1 is header
            if not _truthy(row.get("keep", "")):
                continue
            verbatim = (row.get("verbatim") or "").strip()
            if not verbatim:
                continue
            t = (row.get("type") or "").strip().lower()
            c = (row.get("category") or "").strip().lower()
            if t not in TYPE_ENUM:
                errors.append(f"  row {i}: type '{t}' not in {TYPE_ENUM}")
            if c not in CATEGORY_ENUM:
                errors.append(f"  row {i}: category '{c}' not in {CATEGORY_ENUM}")
            try:
                page = int(row.get("page") or 0)
            except ValueError:
                errors.append(f"  row {i}: page '{row.get('page')}' is not a number")
                page = 0
            due = (row.get("due_date") or "").strip() or None
            reqs.append({
                "id": f"R-{len(reqs) + 1:03d}",
                "verbatim_text": verbatim,
                "section_ref": (row.get("section") or "").strip() or "n/a",
                "type": t,
                "category": c,
                "due_date": due,
                "page": page,
            })
    if errors:
        sys.exit("fix these rows before converting:\n" + "\n".join(errors))
    out = {"rfp_id": rfp_id, "requirements": reqs}
    out_path = GOLD_DIR / f"{rfp_id}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"{rfp_id}: {len(reqs)} kept -> {out_path.relative_to(ROOT)}  (gold answer key)")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--to-gold":
        for rfp_id in args[1:]:
            to_gold(rfp_id)
    else:
        for rfp_id in args:
            build_sheet(rfp_id)


if __name__ == "__main__":
    main()
