"""Tests for the deterministic scorer — the math the whole bake-off rests on.

Run: .venv/bin/python -m pytest evals/ -q
"""

import copy

from score import norm_section, score, token_f1


def gold_fixture():
    return {
        "rfp_id": "TEST-001",
        "requirements": [
            {
                "id": "R-001",
                "verbatim_text": "The vendor shall submit three copies of the proposal.",
                "section_ref": "3.2.1",
                "type": "shall",
                "category": "form",
                "due_date": None,
                "page": 4,
            },
            {
                "id": "R-002",
                "verbatim_text": "Proposals must be received no later than August 1, 2026 at 2:00 PM CT.",
                "section_ref": "1.4",
                "type": "shall",
                "category": "deadline",
                "due_date": "2026-08-01",
                "page": 2,
            },
            {
                "id": "R-003",
                "verbatim_text": "Vendors should attend the pre-proposal conference.",
                "section_ref": "1.5",
                "type": "should",
                "category": "info",
                "due_date": None,
                "page": 2,
            },
        ],
    }


def test_perfect_prediction_scores_one():
    gold = gold_fixture()
    result = score(gold, copy.deepcopy(gold))
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["type_acc"] == 1.0
    assert result["category_acc"] == 1.0
    assert result["due_date_acc"] == 1.0
    assert result["missed_gold_ids"] == []


def test_empty_prediction_scores_zero():
    gold = gold_fixture()
    result = score(gold, {"rfp_id": "TEST-001", "requirements": []})
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["type_acc"] is None
    assert result["missed_gold_ids"] == ["R-001", "R-002", "R-003"]


def test_missing_one_requirement_hits_recall_not_precision():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"] = pred["requirements"][:2]
    result = score(gold, pred)
    assert result["precision"] == 1.0
    assert result["recall"] == round(2 / 3, 3)
    assert result["missed_gold_ids"] == ["R-003"]


def test_invented_requirement_hits_precision_not_recall():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"].append({
        "id": "R-004",
        "verbatim_text": "The vendor shall provide a unicorn.",
        "section_ref": "9.9",
        "type": "shall",
        "category": "deliverable",
        "due_date": None,
        "page": 40,
    })
    result = score(gold, pred)
    assert result["precision"] == 0.75
    assert result["recall"] == 1.0


def test_wrong_section_ref_is_not_a_match():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"][0]["section_ref"] = "4.1"
    result = score(gold, pred)
    assert result["n_matched"] == 2
    assert "R-001" in result["missed_gold_ids"]


def test_section_ref_normalization_ignores_formatting():
    assert norm_section("Section 3.2.1") == norm_section("3.2.1")
    assert norm_section(" 1.4 ") == norm_section("1.4")
    assert norm_section("3.2.1.") == norm_section("3.2.1")


def test_paraphrase_below_threshold_is_not_a_match():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"][0]["verbatim_text"] = "Submit the proposal in triplicate."
    result = score(gold, pred)
    assert "R-001" in result["missed_gold_ids"]


def test_minor_trim_still_matches():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"][0]["verbatim_text"] = (
        "vendor shall submit three copies of the proposal"
    )
    result = score(gold, pred)
    assert result["n_matched"] == 3


def test_gold_row_claimed_only_once():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"].append(copy.deepcopy(pred["requirements"][0]))
    result = score(gold, pred)
    assert result["n_matched"] == 3
    assert result["n_pred"] == 4
    assert result["precision"] == 0.75


def test_wrong_type_and_due_date_tracked_on_matched_rows():
    gold = gold_fixture()
    pred = copy.deepcopy(gold)
    pred["requirements"][1]["type"] = "info"
    pred["requirements"][1]["due_date"] = "2026-08-02"
    result = score(gold, pred)
    assert result["n_matched"] == 3
    assert result["type_acc"] == round(2 / 3, 3)
    assert result["due_date_acc"] == round(2 / 3, 3)


def test_token_f1_bounds():
    assert token_f1("", "anything") == 0.0
    assert token_f1("shall submit", "shall submit") == 1.0
    assert 0.0 < token_f1("the vendor shall submit", "vendor shall submit forms") < 1.0
