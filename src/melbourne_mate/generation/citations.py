"""Check that citations and numbers come from retrieved evidence.

These checks do not prove that an answer is correct. Manual review does that.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_\-]+)\]")
URL_PATTERN = re.compile(r"https?://[^\s<>\]\)\"']+")
# Normalise "$500" and "500" to the same number.
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class CitationReport:
    cited: tuple[str, ...]
    retrieved: tuple[str, ...]
    unsupported: tuple[str, ...]  # cited but not retrieved for this question
    irrelevant: tuple[str, ...]  # retrieved, but qrels say not relevant
    uncited_urls: tuple[str, ...]  # URLs in the prose that no citation backs
    unsupported_numbers: tuple[str, ...]  # figures absent from every cited passage
    has_citation: bool

    @property
    def is_valid(self) -> bool:
        return (
            self.has_citation
            and not self.unsupported
            and not self.irrelevant
            and not self.uncited_urls
            and not self.unsupported_numbers
        )


def extract_citations(answer: str) -> list[str]:
    """Citation ids in order of first appearance."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match not in seen:
            seen.append(match)
    return seen


def _numbers(text: str) -> set[str]:
    stripped = URL_PATTERN.sub(" ", CITATION_PATTERN.sub(" ", text))
    return {match.replace(",", "") for match in NUMBER_PATTERN.findall(stripped)}


def check_citations(
    answer: str,
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str] = (),
    source_urls: Mapping[str, str] | None = None,
    passage_texts: Mapping[str, str] | None = None,
) -> CitationReport:
    """Check one answer against its retrieved and relevant passages."""
    retrieved = tuple(retrieved_ids)
    cited = tuple(extract_citations(answer))
    unsupported = tuple(c for c in cited if c not in retrieved)

    if relevant_ids:
        relevant = set(relevant_ids)
        irrelevant = tuple(c for c in cited if c in retrieved and c not in relevant)
    else:
        irrelevant = ()

    allowed_urls = set()
    if source_urls:
        allowed_urls = {source_urls[c] for c in cited if c in source_urls}
    uncited_urls = tuple(
        url.rstrip(".,;")
        for url in URL_PATTERN.findall(answer)
        if url.rstrip(".,;") not in allowed_urls
    )

    unsupported_numbers: tuple[str, ...] = ()
    if passage_texts:
        cited_numbers: set[str] = set()
        for pid in cited:
            if pid in passage_texts:
                cited_numbers |= _numbers(passage_texts[pid])
        unsupported_numbers = tuple(sorted(_numbers(answer) - cited_numbers))

    return CitationReport(
        cited=cited,
        retrieved=retrieved,
        unsupported=unsupported,
        irrelevant=irrelevant,
        uncited_urls=uncited_urls,
        unsupported_numbers=unsupported_numbers,
        has_citation=bool(cited),
    )
