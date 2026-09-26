"""BM25s lexical baseline for the English test collection."""

from __future__ import annotations

from melbourne_mate.config import CONFIG
from melbourne_mate.retrieval.base import Hit


class BM25Retriever:
    """BM25s lexical retriever."""

    name = "bm25"

    def __init__(
        self,
        passage_ids: list[str],
        texts: list[str],
        k1: float | None = None,
        b: float | None = None,
        stemmer: str | None = None,
    ) -> None:
        if len(passage_ids) != len(texts):
            raise ValueError("passage_ids and texts must be the same length")
        if not passage_ids:
            raise ValueError("cannot index an empty collection")

        import bm25s

        self.passage_ids = list(passage_ids)
        self.k1 = CONFIG.bm25.k1 if k1 is None else k1
        self.b = CONFIG.bm25.b if b is None else b
        self.stemmer_name = CONFIG.bm25.stemmer if stemmer is None else stemmer
        self._stemmer = self._load_stemmer(self.stemmer_name)

        tokens = bm25s.tokenize(
            texts,
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        self._engine = bm25s.BM25(k1=self.k1, b=self.b)
        self._engine.index(tokens, show_progress=False)
        self._bm25s = bm25s

    @staticmethod
    def _load_stemmer(name: str):
        if name in ("", "none"):
            return None
        try:
            import Stemmer

            return Stemmer.Stemmer(name)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyStemmer is required for the frozen BM25s configuration."
            ) from exc

    def search(self, query: str, k: int) -> list[Hit]:
        k = max(1, min(k, len(self.passage_ids)))
        tokens = self._bm25s.tokenize(
            [query],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        indexes, scores = self._engine.retrieve(tokens, k=k, show_progress=False)
        # Drop zero-score documents before assigning ranks.
        scored = [
            (int(index), float(score))
            for index, score in zip(indexes[0], scores[0])
            if float(score) > 0.0
        ]
        return [
            Hit(self.passage_ids[index], rank, score)
            for rank, (index, score) in enumerate(scored, start=1)
        ]
