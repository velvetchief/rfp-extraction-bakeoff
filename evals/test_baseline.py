"""Tests for the lenient JSON recovery + chunking in src/baseline.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baseline import chunk_pages, merge_chunks, parse_matrix_text  # noqa: E402

ROW = ('{"id": "R-001", "verbatim_text": "Vendor shall register.", "section_ref": "1.1", '
       '"type": "shall", "category": "form", "due_date": null, "page": 2}')


def test_parse_clean_object():
    matrix, note = parse_matrix_text(
        '{"rfp_id": "X", "requirements": [' + ROW + "]}", "X")
    assert note == "ok"
    assert len(matrix["requirements"]) == 1


def test_parse_code_fence_and_chatter():
    raw = "Sure! Here is the matrix:\n```json\n{\"requirements\": [" + ROW + "]}\n```\nHope that helps."
    matrix, note = parse_matrix_text(raw, "X")
    assert note == "ok"
    assert matrix["requirements"][0]["id"] == "R-001"


def test_parse_bare_array():
    matrix, note = parse_matrix_text("[" + ROW + "]", "X")
    assert note == "ok"
    assert len(matrix["requirements"]) == 1


def test_parse_truncated_output_recovers_complete_rows():
    raw = '{"requirements": [' + ROW + ", " + ROW.replace("R-001", "R-002") + \
        ', {"id": "R-003", "verbatim_text": "cut off mid senten'
    matrix, note = parse_matrix_text(raw, "X")
    assert note == "repaired truncated JSON"
    assert [r["id"] for r in matrix["requirements"]] == ["R-001", "R-002"]


def test_parse_answer_after_thinking_with_stray_braces():
    raw = ('Let me think. The schema is {"id": ..., "page": N}. Wait, one '
           'check on section {4.2}. OK, final answer:\n'
           '{"rfp_id": "X", "requirements": [' + ROW + "]}")
    matrix, note = parse_matrix_text(raw, "X")
    assert note == "ok"
    assert [r["id"] for r in matrix["requirements"]] == ["R-001"]


def test_parse_truncated_answer_after_thinking():
    raw = ('Thinking about {"braces"} in prose first...\n'
           '{"requirements": [' + ROW + ", " + ROW.replace("R-001", "R-002") +
           ', {"id": "R-003", "verbatim_text": "cut off')
    matrix, note = parse_matrix_text(raw, "X")
    assert note == "repaired truncated JSON"
    assert [r["id"] for r in matrix["requirements"]] == ["R-001", "R-002"]


def test_parse_garbage_scores_zero_not_crash():
    matrix, note = parse_matrix_text("I could not find any requirements.", "X")
    assert matrix["requirements"] == []
    assert note == "no JSON found"


def test_chunks_respect_page_boundaries_and_cover_everything():
    text = "\n".join(
        f"[page {p}]\n" + ("word " * 300).strip() for p in range(1, 8))
    chunks = chunk_pages(text, max_words=1000)
    assert len(chunks) == 3  # 3+3+1 pages at 300 words each
    joined = "\n".join(chunks)
    for p in range(1, 8):
        assert f"[page {p}]" in joined


def test_merge_renumbers_across_chunks():
    a = {"rfp_id": "X", "requirements": [
        {"id": "R-001", "section_ref": "1.1", "verbatim_text": "shall do a"},
        {"id": "R-002", "section_ref": "1.2", "verbatim_text": "shall do b"}]}
    b = {"rfp_id": "X", "requirements": [
        {"id": "R-001", "section_ref": "2.1", "verbatim_text": "shall do c"}]}
    merged = merge_chunks([a, b], "X")
    assert [r["id"] for r in merged["requirements"]] == ["R-001", "R-002", "R-003"]


def test_merge_drops_repeated_rows_but_keeps_section_variants():
    row = {"id": "R-001", "section_ref": "1.1", "verbatim_text": "Shall do A."}
    echo = {"id": "R-002", "section_ref": "1.1)", "verbatim_text": "shall do a"}
    other_section = {"id": "R-003", "section_ref": "9.9",
                     "verbatim_text": "shall do a"}
    merged = merge_chunks(
        [{"rfp_id": "X", "requirements": [row, echo, other_section]}], "X")
    texts = [(r["section_ref"], r["verbatim_text"])
             for r in merged["requirements"]]
    assert texts == [("1.1", "Shall do A."), ("9.9", "shall do a")]
