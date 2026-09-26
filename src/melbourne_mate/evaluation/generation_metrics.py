"""Compute refusal, unsupported-answer, and citation rates without a judge model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from melbourne_mate.evaluation.risk import HIGH_RISK_CATEGORIES as HIGH_RISK


@dataclass(frozen=True)
class GenerationOutcome:
    """Minimal per-question facts needed for deterministic safety metrics."""

    question_id: str
    answerable: bool
    refused: bool
    citations_valid: bool
    retrieval_hit: bool  # a relevant passage was in the context
    truncated: bool = False  # cut off by the token limit: neither answered nor refused
    risk_category: str = "general"
    language: str = "en"
    question_form: str = "canonical"

    @property
    def completed(self) -> bool:
        return not self.truncated


@dataclass(frozen=True)
class GenerationSummary:
    """Aggregate generation and refusal rates for one experimental arm."""

    answerable_n: int
    ookb_n: int
    truncated_n: int
    truncation_rate: float
    correct_refusal_rate: float
    false_refusal_rate: float
    unsupported_answer_rate: float
    citation_validity_rate: float
    high_risk_unsupported_rate: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "answerable_n": self.answerable_n,
            "ookb_n": self.ookb_n,
            "truncated_n": self.truncated_n,
            "truncation_rate": self.truncation_rate,
            "correct_refusal_rate": self.correct_refusal_rate,
            "false_refusal_rate": self.false_refusal_rate,
            "unsupported_answer_rate": self.unsupported_answer_rate,
            "citation_validity_rate": self.citation_validity_rate,
            "high_risk_unsupported_rate": self.high_risk_unsupported_rate,
        }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarise_generation(outcomes: Sequence[GenerationOutcome]) -> GenerationSummary:
    # Report truncated output separately from answers and refusals.
    truncated = [o for o in outcomes if o.truncated]
    complete = [o for o in outcomes if o.completed]
    answerable = [o for o in complete if o.answerable]
    ookb = [o for o in complete if not o.answerable]
    answered_ookb = [o for o in ookb if not o.refused]
    answered_answerable = [o for o in answerable if not o.refused]

    # Count false refusals only when the needed evidence was retrieved.
    refusable = [o for o in answerable if o.retrieval_hit]

    return GenerationSummary(
        answerable_n=len(answerable),
        ookb_n=len(ookb),
        truncated_n=len(truncated),
        truncation_rate=_rate(len(truncated), len(outcomes)),
        correct_refusal_rate=_rate(sum(1 for o in ookb if o.refused), len(ookb)),
        false_refusal_rate=_rate(sum(1 for o in refusable if o.refused), len(refusable)),
        unsupported_answer_rate=_rate(len(answered_ookb), len(ookb)),
        citation_validity_rate=_rate(
            sum(1 for o in answered_answerable if o.citations_valid),
            len(answered_answerable),
        ),
        high_risk_unsupported_rate=(
            _rate(
                sum(1 for o in answered_ookb if o.risk_category in HIGH_RISK),
                len([o for o in ookb if o.risk_category in HIGH_RISK]),
            )
            if any(o.risk_category in HIGH_RISK for o in ookb)
            else None
        ),
    )


def contingency_2x2(outcomes: Sequence[GenerationOutcome]) -> dict[str, int]:
    """Cross retrieval success with the answer outcome."""
    answerable = [o for o in outcomes if o.answerable and o.completed]
    return {
        "hit_answered": sum(1 for o in answerable if o.retrieval_hit and not o.refused),
        "hit_refused": sum(1 for o in answerable if o.retrieval_hit and o.refused),
        "miss_answered": sum(1 for o in answerable if not o.retrieval_hit and not o.refused),
        "miss_refused": sum(1 for o in answerable if not o.retrieval_hit and o.refused),
    }
