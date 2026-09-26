"""Run retrieval, evidence checks, generation, and citation checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import Collection
from melbourne_mate.generation.citations import CitationReport, check_citations
from melbourne_mate.generation.contracts import GenerationRequest, LanguageModel
from melbourne_mate.generation.gate import EvidenceGate, GateDecision
from melbourne_mate.generation.prompts import (
    FALLBACK_TEXT,
    EvidenceItem,
    build_closed_book_prompt,
    build_prompt,
)
from melbourne_mate.retrieval.base import Hit, Retriever
from melbourne_mate.retrieval.bm25 import BM25Retriever
from melbourne_mate.retrieval.dense import DenseRetriever

ARMS = ("bm25", "dense", "none")


# Providers use these values when output stops at the token limit.
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "maxtokens", "truncated"})


@dataclass(frozen=True)
class Answer:
    """Output from one pipeline request."""

    question: str
    status: str  # "answered" | "refused" | "truncated"
    text: str
    hits: tuple[Hit, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    gate: GateDecision | None = None
    citations: CitationReport | None = None
    latency_ms: int = 0
    arm: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    @property
    def truncated(self) -> bool:
        return self.status == "truncated"


def build_retriever(collection: Collection, arm: str, encoder=None) -> Retriever | None:
    """Build one retrieval arm; `none` is the no-context ablation."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if arm == "none":
        return None

    pids = sorted(collection.passages)
    texts = [collection.indexing_text(pid) for pid in pids]

    if arm == "bm25":
        return BM25Retriever(pids, texts)
    if encoder is None:
        raise ValueError(f"arm {arm!r} needs a dense encoder")
    return DenseRetriever(pids, texts, encoder)


class RagPipeline:
    """Shared orchestration used by both the evaluated runs and the demo UI."""

    def __init__(
        self,
        collection: Collection,
        retriever: Retriever | None,
        model: LanguageModel,
        gate: EvidenceGate | None = None,
        arm: str = "",
        top_k: int | None = None,
    ) -> None:
        self.collection = collection
        self.retriever = retriever
        self.model = model
        self.gate = gate
        self.arm = arm or (retriever.name if retriever else "none")
        self.top_k = CONFIG.final_top_k if top_k is None else top_k

    def _evidence(self, hits: tuple[Hit, ...]) -> tuple[EvidenceItem, ...]:
        items = []
        for hit in hits:
            passage = self.collection.passages[hit.passage_id]
            source = self.collection.sources[passage.source_id]
            items.append(
                EvidenceItem(
                    passage_id=passage.passage_id,
                    heading=passage.section_heading,
                    text=passage.text,
                    organisation=source.organisation,
                    url=source.url,
                )
            )
        return tuple(items)

    def retrieve(self, question: str) -> tuple[Hit, ...]:
        if self.retriever is None:
            return ()
        return tuple(self.retriever.search(question, self.top_k))

    def answer(
        self,
        question: str,
        relevant_ids: tuple[str, ...] = (),
        on_chunk: Callable[[str], None] | None = None,
    ) -> Answer:
        """Retrieve evidence, apply the gate, generate, and check citations."""
        hits = self.retrieve(question)
        evidence = self._evidence(hits)

        decision = None
        if self.gate is not None and self.retriever is not None:
            decision = self.gate.decide(question, [item.text for item in evidence])
            if not decision.sufficient:
                return Answer(
                    question=question,
                    status="refused",
                    text=FALLBACK_TEXT,
                    hits=hits,
                    evidence=evidence,
                    gate=decision,
                    arm=self.arm,
                )

        # No-context needs its own prompt.
        prompt = (
            build_closed_book_prompt(question)
            if self.retriever is None
            else build_prompt(question, evidence)
        )
        req = GenerationRequest(
            prompt=prompt, temperature=CONFIG.generation.temperature
        )
        streamer = getattr(self.model, "stream", None)
        if on_chunk is not None and callable(streamer):
            for piece in streamer(req):
                on_chunk(piece)
            resp = getattr(self.model, "last_response", None)
            if resp is None:  # streamers must save the final response
                raise RuntimeError(
                    f"{type(self.model).__name__}.stream did not record a response"
                )
        else:
            resp = self.model.generate(req)
        text = resp.text.strip()
        if _finish_reason_is_truncation(resp.finish_reason):
            status = "truncated"
        elif _is_fallback(text):
            status = "refused"
        else:
            status = "answered"

        source_urls = {item.passage_id: item.url for item in evidence}
        report = check_citations(
            text,
            retrieved_ids=[item.passage_id for item in evidence],
            relevant_ids=relevant_ids,
            source_urls=source_urls,
            passage_texts={item.passage_id: item.text for item in evidence},
        )
        return Answer(
            question=question,
            status=status,
            text=text,
            hits=hits,
            evidence=evidence,
            gate=decision,
            citations=report,
            latency_ms=resp.latency_ms,
            arm=self.arm,
            meta={
                "model": resp.model,
                "prompt_version": CONFIG.generation.prompt_version,
            },
        )


def _finish_reason_is_truncation(reason: str) -> bool:
    """Normalise provider finish reasons."""
    return reason.strip().lower().replace("_", "").replace("-", "") in {
        r.replace("_", "") for r in TRUNCATION_REASONS
    }


def _is_fallback(text: str) -> bool:
    """Allow small formatting changes in the fixed fallback."""
    normalised = " ".join(text.lower().split())
    marker = " ".join(FALLBACK_TEXT.lower().split())[:48]
    return marker in normalised
