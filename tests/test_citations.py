from melbourne_mate.generation.citations import check_citations, extract_citations


def test_extracts_multiple_ids_from_one_bracket():
    answer = "Use both sources [P04-2, P30-1]."

    assert extract_citations(answer) == ["P04-2", "P30-1"]


def test_grouped_citation_ids_are_not_treated_as_numbers():
    report = check_citations(
        "Use both sources [P04-2, P30-1].",
        retrieved_ids=("P04-2", "P30-1"),
        relevant_ids=("P04-2", "P30-1"),
        passage_texts={"P04-2": "First source.", "P30-1": "Second source."},
    )

    assert report.is_valid
    assert report.unsupported_numbers == ()
