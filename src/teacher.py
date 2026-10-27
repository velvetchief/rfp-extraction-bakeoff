"""Teacher pass: Claude reads an RFP PDF and drafts the compliance matrix.

Output goes to data/train/<rfp_id>.json for human verification before it
becomes training data. Gold RFPs (data/gold/) never come through here.

Usage: python src/teacher.py data/rfps/HHS0015558.pdf [rfp_id]
       (rfp_id defaults to the PDF filename stem)
"""

import base64
import json
import sys
from pathlib import Path

from anthropic import Anthropic

from prompts import EXTRACTION_SYSTEM, EXTRACTION_USER

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-4-8"


def strict(node):
    """Recursively add additionalProperties: false to every object schema,
    as required by the structured-outputs API."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
        for v in node.values():
            strict(v)
    elif isinstance(node, list):
        for v in node:
            strict(v)
    return node


def extract(pdf_path: Path, rfp_id: str) -> dict:
    schema = strict(json.loads((ROOT / "schema.json").read_text()))
    pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode()

    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system=EXTRACTION_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": EXTRACTION_USER.format(rfp_id=rfp_id)},
            ],
        }],
    ) as stream:
        message = stream.get_final_message()

    text = "".join(b.text for b in message.content if b.type == "text")
    matrix = json.loads(text)
    matrix["rfp_id"] = rfp_id  # trust the CLI arg, not the model
    return matrix


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pdf_path = Path(sys.argv[1])
    rfp_id = sys.argv[2] if len(sys.argv) > 2 else pdf_path.stem
    matrix = extract(pdf_path, rfp_id)

    out = ROOT / "data" / "train" / f"{rfp_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2))
    n = len(matrix["requirements"])
    print(f"{rfp_id}: {n} requirements -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
