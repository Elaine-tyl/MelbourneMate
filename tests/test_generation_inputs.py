import csv

import pytest

from melbourne_mate.corpus import RISK_CATEGORIES, load_collection
from melbourne_mate.evaluation.generation_inputs import (
    GenerationInputError,
    generation_sample_fingerprint,
    load_generation_cases,
    load_generation_qrels,
)


def write_sample(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "risk_category"])
        writer.writeheader()
        writer.writerows(rows)


def test_formal_sample_has_the_frozen_question_mix():
    collection = load_collection("data/v1")
    cases = load_generation_cases("data/v1/generation-sample.csv", collection)

    assert len(cases) == 66
    assert len({case.question_id for case in cases}) == 66
    assert sum(collection.questions[case.question_id].is_ookb for case in cases) == 30
    assert sum(not collection.questions[case.question_id].is_ookb for case in cases) == 36
    assert {case.risk_category for case in cases} <= RISK_CATEGORIES
    assert len(generation_sample_fingerprint(cases)) == 16


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"question_id": "Q04C", "risk_category": "employment"},
                {"question_id": "Q04C", "risk_category": "employment"},
            ],
            "duplicate question_id Q04C",
        ),
        (
            [{"question_id": "Q404", "risk_category": "general"}],
            "unknown question_id Q404",
        ),
        (
            [{"question_id": "Q04C", "risk_category": "finance"}],
            "invalid risk_category finance",
        ),
    ],
)
def test_sample_rejects_bad_rows(tmp_path, rows, message):
    path = tmp_path / "sample.csv"
    write_sample(path, rows)

    with pytest.raises(GenerationInputError, match=message):
        load_generation_cases(path, load_collection("data/v1"))


def test_final_qrels_load_with_a_stable_fingerprint():
    collection = load_collection("data/v1")

    qrels, fingerprint = load_generation_qrels(
        "review/targeted-qrels/qrels-final.txt", collection
    )

    # Guard the reviewed relevance-pair total against accidental data loss.
    assert sum(len(passages) for passages in qrels.values()) == 71
    assert len(fingerprint) == 16


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("Q404 0 P04-1 2\n", "unknown question_id Q404"),
        ("Q04C 0 P404 2\n", "unknown passage_id P404"),
    ],
)
def test_qrels_reject_unknown_ids(tmp_path, line, message):
    path = tmp_path / "qrels.txt"
    path.write_text(line, encoding="utf-8")

    with pytest.raises(GenerationInputError, match=message):
        load_generation_qrels(path, load_collection("data/v1"))
