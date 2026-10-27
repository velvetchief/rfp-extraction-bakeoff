import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from label_aid import load_blocks  # noqa: E402

DOC = """[page 1]
2.2.1 Comprehensive Services
The Respondent shall do the following:
1. Educate potential vendors on program participation
2. Assist vendors in soliciting contracts
Mail to 1100 West 49th Street, Austin.
2.2.2 Key Personnel
The Respondent must staff the account.
"""


def sections(strict):
    return {s: sec for _, sec, s in load_blocks(DOC, strict_sections=strict)}


def section_of(secs, fragment):
    return next(v for k, v in secs.items() if fragment in k)


def test_strict_keeps_list_items_in_parent_section():
    secs = sections(True)
    assert section_of(secs, "Assist vendors") == "2.2.1"
    assert section_of(secs, "1100 West 49th") == "2.2.1"
    assert section_of(secs, "must staff the account") == "2.2.2"


def test_default_behavior_unchanged_for_gold_reproducibility():
    assert "1" in set(sections(False).values())
