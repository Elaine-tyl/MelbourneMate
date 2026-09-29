import numpy as np
import pytest

from melbourne_mate.retrieval import BM25Retriever, DenseRetriever, Retriever
from melbourne_mate.retrieval.encoders import HashingEncoder

IDS = ["P1", "P2", "P3"]
TEXTS = [
    "Student visa holders may work 48 hours a fortnight during the course.",
    "The rental bond is lodged with the RTBA within 14 days.",
    "Call Triple Zero in a medical emergency.",
]


class FixedEncoder:
    """Returns set vectors so dense ranking can be checked exactly."""

    def __init__(self, passages, query):
        self.passages = np.asarray(passages, dtype=np.float32)
        self.query = np.asarray([query], dtype=np.float32)

    def encode_passages(self, texts):
        return self.passages

    def encode_queries(self, texts):
        return self.query


def test_bm25_ranks_the_matching_passage_first():
    hits = BM25Retriever(IDS, TEXTS).search("Where is my rental bond lodged?", k=3)

    assert hits[0].passage_id == "P2"
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_bm25_uses_the_stemmer_for_word_forms():
    hits = BM25Retriever(IDS, TEXTS).search("worked", k=1)
    unstemmed = BM25Retriever(IDS, TEXTS, stemmer="none").search("worked", k=1)

    assert hits[0].passage_id == "P1"
    assert unstemmed == []


def test_bm25_drops_passages_with_no_shared_terms():
    hits = BM25Retriever(IDS, TEXTS).search("rental bond", k=3)

    assert [hit.passage_id for hit in hits] == ["P2"]


def test_bm25_limits_k_to_the_collection_size():
    hits = BM25Retriever(IDS, TEXTS).search("student course bond emergency", k=10)

    assert sorted(hit.passage_id for hit in hits) == IDS
    assert [hit.rank for hit in hits] == [1, 2, 3]


@pytest.mark.parametrize(
    ("ids", "texts", "message"),
    [
        (["P1"], ["one", "two"], "same length"),
        ([], [], "empty collection"),
    ],
)
def test_bm25_rejects_bad_input(ids, texts, message):
    with pytest.raises(ValueError, match=message):
        BM25Retriever(ids, texts)


def test_dense_ranks_by_similarity_and_breaks_ties_by_passage_id():
    encoder = FixedEncoder([[0.6, 0.8], [1.0, 0.0], [1.0, 0.0]], [1.0, 0.0])
    hits = DenseRetriever(["P3", "P2", "P1"], ["a", "b", "c"], encoder).search("q", k=3)

    assert [hit.passage_id for hit in hits] == ["P1", "P2", "P3"]
    assert hits[0].score == pytest.approx(1.0)


def test_dense_with_hashing_encoder_finds_the_closest_text():
    retriever = DenseRetriever(IDS, TEXTS, HashingEncoder())
    hits = retriever.search("medical emergency Triple Zero", k=2)

    assert isinstance(retriever, Retriever)
    assert hits[0].passage_id == "P3"
    assert len(hits) == 2


def test_dense_rejects_a_wrong_number_of_passage_vectors():
    encoder = FixedEncoder([[1.0, 0.0]], [1.0, 0.0])

    with pytest.raises(ValueError, match="wrong number"):
        DenseRetriever(IDS, TEXTS, encoder)
