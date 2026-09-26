"""Prompt construction, frozen as prompt version v1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

PROMPT_VERSION = "v1"

FALLBACK_TEXT = (
    "I don't have approved information that answers this. "
    "Please check the official source directly or contact your university's "
    "international student support team."
)

SYSTEM_RULES = """You are MelbourneMate, an assistant for international students settling in Melbourne.

Rules:
1. Answer only from the numbered evidence below. Never use outside knowledge.
2. Cite the evidence id in square brackets after each factual claim, e.g. [P012].
3. Copy amounts, dates, deadlines, conditions and organisation names exactly as
   they appear in the evidence. Never round, convert or paraphrase a number.
4. If the evidence does not answer the question, reply with exactly:
   {fallback}
5. Reply in clear English. Keep official names and URLs unchanged.
6. Keep the answer to 3-6 sentences.
"""


@dataclass(frozen=True)
class EvidenceItem:
    passage_id: str
    heading: str
    text: str
    organisation: str
    url: str


def quote_evidence(text: str) -> str:
    """Remove prompt delimiters found inside collected text."""
    return text.replace("<<<", "").replace(">>>", "").strip()


CLOSED_BOOK_RULES = """You are an assistant helping international students settling in Melbourne.

Answer the question from your own knowledge. Keep the answer to 3-6 sentences.
If you are not confident, say so rather than guessing.
"""


def build_closed_book_prompt(question: str) -> str:
    """Build the no-context baseline prompt without forced refusal."""
    return f"{CLOSED_BOOK_RULES}\nStudent question: {question}\nAnswer:"


def build_prompt(question: str, evidence: Sequence[EvidenceItem]) -> str:
    """Grounded prompt. Requires evidence; the ablation uses the closed-book one."""
    blocks = []
    for item in evidence:
        blocks.append(
            f"[{item.passage_id}] {item.organisation} — {item.heading}\n"
            f"<<<{quote_evidence(item.text)}>>>\n"
            f"Source: {item.url}"
        )
    evidence_block = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    # Grounded arms must refuse when retrieval returns no evidence.
    return (
        SYSTEM_RULES.format(fallback=FALLBACK_TEXT)
        + "\nEvidence:\n"
        + evidence_block
        + f"\n\nStudent question: {question}\nAnswer:"
    )
