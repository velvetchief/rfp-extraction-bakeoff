"""LLM judge layer: Claude and Grok independently grade a predicted matrix.

Judges see the RFP text (page-marked, from pdftext.py) and the predicted
matrix. They grade dimensions the deterministic scorer can't: coverage of the
whole document, faithfulness (no invented or paraphrased requirements), and
usability as a working compliance matrix. The two judges are ALWAYS reported
separately — never averaged.

Requires ANTHROPIC_API_KEY; Grok judge also needs XAI_API_KEY (skipped with a
warning if absent). GROK_MODEL env var overrides the default model name.

Usage: python src/judge.py data/rfps/HHS0015558.txt predicted.json
"""

import json
import os
import sys
from pathlib import Path

import httpx
from anthropic import Anthropic

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["coverage", "faithfulness", "usability", "problems", "rationale"],
    "properties": {
        "coverage": {
            "type": "integer",
            "description": "Did the matrix capture every requirement in the RFP? 10 = nothing material missed.",
        },
        "faithfulness": {
            "type": "integer",
            "description": "Is every row real and verbatim? 10 = no invented, duplicated, or paraphrased rows.",
        },
        "usability": {
            "type": "integer",
            "description": "Could a proposal manager work directly from this matrix? Sections, types, dates correct.",
        },
        "problems": {
            "type": "array", "items": {"type": "string"},
            "description": "Specific defects found, worst first, max 10.",
        },
        "rationale": {"type": "string", "description": "2-3 sentence overall judgment."},
    },
}

JUDGE_SYSTEM = """\
You are grading a compliance matrix that a model extracted from a government \
RFP. You have the full RFP text (with [page N] markers) and the extracted \
matrix. Grade strictly:

- coverage (0-10): scan the RFP for shall/must/should statements, deadlines, \
required forms, and evaluation criteria. Every material one missing from the \
matrix costs points.
- faithfulness (0-10): spot-check rows against the RFP text. Invented \
requirements, paraphrased (non-verbatim) text, and wrong section references \
cost points heavily.
- usability (0-10): correct type/category tags, ISO due dates, accurate page \
numbers, one obligation per row.

List concrete problems with section references. Do not grade on style.\
"""

JUDGE_USER = """\
## RFP text
{rfp_text}

## Extracted compliance matrix
{matrix_json}

Grade the matrix against the RFP.\
"""


def judge_claude(rfp_text: str, matrix: dict) -> dict:
    client = Anthropic()
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=JUDGE_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": JUDGE_USER.format(
                rfp_text=rfp_text, matrix_json=json.dumps(matrix, indent=2)
            ),
        }],
    ) as stream:
        message = stream.get_final_message()
    text = "".join(b.text for b in message.content if b.type == "text")
    return json.loads(text)


def judge_grok(rfp_text: str, matrix: dict) -> dict | None:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        print("WARNING: XAI_API_KEY not set — skipping Grok judge", file=sys.stderr)
        return None
    model = os.environ.get("GROK_MODEL", "grok-4.5")
    schema_note = (
        "Respond with ONLY a JSON object matching this schema:\n"
        + json.dumps(JUDGE_SCHEMA, indent=2)
    )
    resp = httpx.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM + "\n\n" + schema_note},
                {"role": "user", "content": JUDGE_USER.format(
                    rfp_text=rfp_text, matrix_json=json.dumps(matrix, indent=2)
                )},
            ],
        },
        timeout=600,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    rfp_text = Path(sys.argv[1]).read_text()
    matrix = json.loads(Path(sys.argv[2]).read_text())
    verdicts = {
        "claude": judge_claude(rfp_text, matrix),
        "grok": judge_grok(rfp_text, matrix),
    }
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
