"""Measure source coverage, review quality, and question wording difficulty."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from melbourne_mate.corpus import Collection, Judgement
from melbourne_mate.text import containment, jaccard, terms


@dataclass(frozen=True)
class Agreement:
    overlapping_items: int
    exact_match: float
    kappa: float
    judges: tuple[str, ...]


@dataclass(frozen=True)
class QualityReport:
    sources: int
    passages: int
    topics: int
    questions_by_form: dict[str, int]
    questions_by_language: dict[str, int]
    inferred_topics: int
    volatile_passages: int
    judged_pairs: int
    double_judged_share: float
    agreement: Agreement | None
    gold_coverage: float
    mean_support_spans: float
    confusable_pairs: int
    lexical_overlap: dict[str, float] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [
            f"sources: {self.sources}",
            f"passages: {self.passages} ({self.volatile_passages} marked volatile)",
            f"topics: {self.topics} ({self.inferred_topics} inferred)",
            "questions: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.questions_by_form.items())),
            "languages: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.questions_by_language.items())),
            f"judged (question, passage) pairs: {self.judged_pairs}",
            f"double-judged share: {self.double_judged_share:.1%}",
        ]
        if self.agreement:
            out.append(
                f"agreement on {self.agreement.overlapping_items} overlapping items: "
                f"exact {self.agreement.exact_match:.1%}, Cohen's kappa "
                f"{self.agreement.kappa:.3f} ({' vs '.join(self.agreement.judges)})"
            )
        gold_line = (
            f"gold answers: {self.gold_coverage:.1%} of answerable questions, "
            f"{self.mean_support_spans:.1f} support sentences each"
        )
        out += [
            gold_line,
            f"confusable topic pairs: {self.confusable_pairs}",
            "question-term containment in the gold passage (lower = harder): "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.lexical_overlap.items())),
        ]
        return out


def cohens_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    """Cohen's kappa for two annotators over the same items."""
    if len(left) != len(right):
        raise ValueError("both judges must cover the same items")
    n = len(left)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(left, right) if a == b) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / n) * (right_counts[label] / n)
        for label in set(left) | set(right)
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def weighted_cohens_kappa(
    left: Sequence[int],
    right: Sequence[int],
    labels: Sequence[int] = (0, 1, 2),
) -> float:
    """Quadratic-weighted Cohen's kappa for ordered scores."""
    if len(left) != len(right):
        raise ValueError("both judges must cover the same items")
    if not left:
        return 0.0
    allowed = tuple(labels)
    if len(allowed) < 2 or len(set(allowed)) != len(allowed):
        raise ValueError("labels must contain at least two unique scores")
    if any(score not in allowed for score in (*left, *right)):
        raise ValueError("scores must be present in labels")

    positions = {label: index for index, label in enumerate(allowed)}
    scale = (len(allowed) - 1) ** 2
    n = len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)

    observed = sum(
        (positions[a] - positions[b]) ** 2 / scale for a, b in zip(left, right)
    ) / n
    expected = sum(
        (left_counts[a] / n)
        * (right_counts[b] / n)
        * ((positions[a] - positions[b]) ** 2 / scale)
        for a in allowed
        for b in allowed
    )
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def _agreement(judgements: Sequence[Judgement]) -> tuple[Agreement | None, int, float]:
    by_item: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for judgement in judgements:
        by_item[(judgement.question_id, judgement.passage_id)][judgement.judge] = judgement.grade

    judged_pairs = len(by_item)
    if not judged_pairs:
        return None, 0, 0.0

    multi = {item: grades for item, grades in by_item.items() if len(grades) > 1}
    double_share = len(multi) / judged_pairs
    if not multi:
        return None, judged_pairs, 0.0

    judges = Counter(judge for grades in multi.values() for judge in grades)
    top_two = tuple(judge for judge, _ in judges.most_common(2))
    if len(top_two) < 2:
        return None, judged_pairs, double_share

    left, right = [], []
    for grades in multi.values():
        if top_two[0] in grades and top_two[1] in grades:
            left.append(grades[top_two[0]])
            right.append(grades[top_two[1]])
    if not left:
        return None, judged_pairs, double_share

    matches = sum(1 for a, b in zip(left, right) if a == b) / len(left)
    return (
        Agreement(len(left), matches, cohens_kappa(left, right), top_two),
        judged_pairs,
        double_share,
    )


CONTAINMENT_BINS = ((0.4, "low"), (0.7, "medium"))


def containment_bin(value: float) -> str:
    """Group a question by how much wording it shares with the gold passage."""
    for threshold, label in CONTAINMENT_BINS:
        if value < threshold:
            return label
    return "high"


def question_containment(collection: Collection) -> dict[str, float]:
    """Per-question term containment in its fully-relevant passages."""
    out: dict[str, float] = {}
    for question in collection.questions.values():
        if question.is_ookb:
            continue
        relevant = [
            passage_id
            for passage_id, grade in collection.qrels.get(question.question_id, {}).items()
            if grade == 2
        ]
        if not relevant:
            continue
        passage_terms: set[str] = set()
        for passage_id in relevant:
            passage = collection.passages[passage_id]
            passage_terms |= terms(f"{passage.section_heading} {passage.text}")
        out[question.question_id] = containment(terms(question.text), passage_terms)
    return out


def lexical_overlap(collection: Collection, measure: str = "containment") -> dict[str, float]:
    """Calculate mean question-to-gold overlap for each question form."""
    if measure not in ("containment", "jaccard"):
        raise ValueError("measure must be 'containment' or 'jaccard'")

    grouped: dict[str, list[float]] = defaultdict(list)
    for question in collection.questions.values():
        if question.is_ookb:
            continue
        relevant = [
            passage_id
            for passage_id, grade in collection.qrels.get(question.question_id, {}).items()
            if grade == 2
        ]
        if not relevant:
            continue
        passage_terms: set[str] = set()
        for passage_id in relevant:
            passage = collection.passages[passage_id]
            passage_terms |= terms(f"{passage.section_heading} {passage.text}")
        question_terms = terms(question.text)
        value = (
            containment(question_terms, passage_terms)
            if measure == "containment"
            else jaccard(question_terms, passage_terms)
        )
        grouped[question.question_form].append(value)

    report = {
        form: sum(values) / len(values) for form, values in grouped.items() if values
    }
    everything = [v for values in grouped.values() for v in values]
    if everything:
        report["all"] = sum(everything) / len(everything)
    return report


def build_report(collection: Collection) -> QualityReport:
    answerable = [q for q in collection.questions.values() if not q.is_ookb]
    agreement, judged_pairs, double_share = _agreement(collection.judgements)

    gold_for_answerable = [
        collection.gold[q.question_id] for q in answerable if q.question_id in collection.gold
    ]
    spans = [len(gold.support) for gold in gold_for_answerable]

    return QualityReport(
        sources=len(collection.sources),
        passages=len(collection.passages),
        topics=len(collection.topics),
        questions_by_form=dict(
            Counter(q.question_form for q in collection.questions.values())
        ),
        questions_by_language=dict(
            Counter(q.language for q in collection.questions.values())
        ),
        inferred_topics=sum(
            1 for t in collection.topics.values() if t.knowledge_type == "inferred"
        ),
        volatile_passages=sum(
            1 for p in collection.passages.values() if p.volatility == "volatile"
        ),
        judged_pairs=judged_pairs or sum(len(v) for v in collection.qrels.values()),
        double_judged_share=double_share,
        agreement=agreement,
        gold_coverage=len(gold_for_answerable) / len(answerable) if answerable else 0.0,
        mean_support_spans=sum(spans) / len(spans) if spans else 0.0,
        confusable_pairs=len(collection.pairs),
        lexical_overlap=lexical_overlap(collection),
    )
