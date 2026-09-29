"""Load the test collection and reject invalid ids, labels, or split links."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_FORMS = {"canonical", "paraphrased", "ookb"}
KNOWLEDGE_TYPES = {"known", "inferred"}
SPLITS = {"validation", "test"}
RISK_CATEGORIES = {"general", "visa", "health", "employment", "housing", "emergency"}
VOLATILITY = {"stable", "volatile"}
MIN_PASSAGE_WORDS = 40
MAX_PASSAGE_WORDS = 300

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sentences(text: str) -> list[str]:
    """Split text in a stable way so support indexes do not move."""
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]


class CollectionError(ValueError):
    """Raised when the collection violates a structural rule."""


@dataclass(frozen=True)
class Source:
    source_id: str
    organisation: str
    title: str
    url: str
    access_date: str
    risk_category: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    topic_id: str
    source_id: str
    section_heading: str
    text: str
    volatility: str = "stable"  # "volatile" = holds a figure that changes

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def sentences(self) -> list[str]:
        return sentences(self.text)


@dataclass(frozen=True)
class Topic:
    topic_id: str
    category: str
    knowledge_type: str
    information_need: str


@dataclass(frozen=True)
class Question:
    question_id: str
    topic_id: str  # empty for out-of-knowledge-base questions
    question_form: str
    language: str
    text: str

    @property
    def is_ookb(self) -> bool:
        return self.question_form == "ookb"


@dataclass(frozen=True)
class Judgement:
    """One annotator's grade for one (question, passage) pair."""

    question_id: str
    passage_id: str
    grade: int
    judge: str
    note: str = ""


@dataclass(frozen=True)
class GoldAnswer:
    """A reference answer and the sentences that support it."""

    question_id: str
    answer: str
    support: tuple[tuple[str, int], ...]  # (passage_id, sentence index)


@dataclass(frozen=True)
class ConfusablePair:
    """Two similar topics that a student or retriever may confuse."""

    topic_a: str
    topic_b: str
    reason: str


@dataclass(frozen=True)
class Collection:
    """Validated in-memory representation of the complete test collection."""

    sources: dict[str, Source]
    passages: dict[str, Passage]
    topics: dict[str, Topic]
    questions: dict[str, Question]
    qrels: dict[str, dict[str, int]]  # question_id -> passage_id -> grade
    splits: dict[str, str]  # topic_id -> split
    judgements: tuple[Judgement, ...] = ()
    gold: dict[str, GoldAnswer] = field(default_factory=dict)
    pairs: tuple[ConfusablePair, ...] = ()

    def indexing_text(self, passage_id: str) -> str:
        """Return the topic, heading, and body indexed by both retrievers."""
        passage = self.passages[passage_id]
        topic = self.topics[passage.topic_id]
        return f"{topic.category} | {passage.section_heading} | {passage.text}"

    def questions_in_split(self, split: str) -> list[Question]:
        """Answerable questions whose topic is in `split`, id-sorted."""
        return sorted(
            (
                q
                for q in self.questions.values()
                if not q.is_ookb and self.splits.get(q.topic_id) == split
            ),
            key=lambda q: q.question_id,
        )

    def ookb_questions(self) -> list[Question]:
        return sorted(
            (q for q in self.questions.values() if q.is_ookb),
            key=lambda q: q.question_id,
        )

    def fingerprint(self) -> str:
        """Content hash of everything a metric could depend on."""
        digest = hashlib.sha256()
        for pid in sorted(self.passages):
            digest.update(pid.encode())
            digest.update(self.indexing_text(pid).encode())
        for qid in sorted(self.questions):
            question = self.questions[qid]
            digest.update(qid.encode())
            digest.update(question.text.encode())
            for pid, grade in sorted(self.qrels.get(qid, {}).items()):
                digest.update(f"{pid}:{grade}".encode())
        for topic_id in sorted(self.splits):
            digest.update(f"{topic_id}:{self.splits[topic_id]}".encode())
        return digest.hexdigest()[:16]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CollectionError(f"missing file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)
        ]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """TREC-style qrels: `question_id 0 passage_id grade`, grades 0-2."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise CollectionError(f"{path}:{line_no}: expected 4 fields, got {len(parts)}")
            question_id, _, passage_id, grade = parts
            try:
                value = int(grade)
            except ValueError as exc:
                raise CollectionError(f"{path}:{line_no}: grade must be an integer") from exc
            if value not in (0, 1, 2):
                raise CollectionError(f"{path}:{line_no}: grade must be 0, 1 or 2")
            if value:
                qrels[question_id][passage_id] = value
    return dict(qrels)


def load_judgements(path: Path) -> tuple[Judgement, ...]:
    """Per-annotator grades. Optional: a collection can ship only qrels."""
    if not path.exists():
        return ()
    out = []
    for row in _read_csv(path):
        try:
            grade = int(row["grade"])
        except (KeyError, ValueError) as exc:
            raise CollectionError(f"{path}: grade must be an integer") from exc
        out.append(
            Judgement(
                question_id=row["question_id"],
                passage_id=row["passage_id"],
                grade=grade,
                judge=row["judge"],
                note=row.get("note", ""),
            )
        )
    return tuple(out)


def _parse_support(value: str) -> tuple[tuple[str, int], ...]:
    """`P001#0;P004#2` -> (("P001", 0), ("P004", 2))."""
    spans = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "#" not in chunk:
            raise CollectionError(f"support span {chunk!r} must look like P001#0")
        passage_id, index = chunk.rsplit("#", 1)
        try:
            spans.append((passage_id.strip(), int(index)))
        except ValueError as exc:
            raise CollectionError(f"support span {chunk!r} has a non-numeric index") from exc
    return tuple(spans)


def load_gold(path: Path) -> dict[str, GoldAnswer]:
    if not path.exists():
        return {}
    return {
        row["question_id"]: GoldAnswer(
            question_id=row["question_id"],
            answer=row["answer"],
            support=_parse_support(row.get("support", "")),
        )
        for row in _read_csv(path)
    }


def load_pairs(path: Path) -> tuple[ConfusablePair, ...]:
    if not path.exists():
        return ()
    return tuple(
        ConfusablePair(row["topic_a"], row["topic_b"], row.get("reason", ""))
        for row in _read_csv(path)
    )


def load_collection(data_dir: str | Path) -> Collection:
    """Load and validate all required and optional collection files."""
    root = Path(data_dir)

    sources = {
        row["source_id"]: Source(
            source_id=row["source_id"],
            organisation=row["organisation"],
            title=row["title"],
            url=row["url"],
            access_date=row["access_date"],
            risk_category=row["risk_category"],
        )
        for row in _read_csv(root / "sources.csv")
    }
    passages = {
        row["passage_id"]: Passage(
            passage_id=row["passage_id"],
            topic_id=row["topic_id"],
            source_id=row["source_id"],
            section_heading=row["section_heading"],
            text=row["text"],
            volatility=row.get("volatility") or "stable",
        )
        for row in _read_csv(root / "passages.csv")
    }
    topics = {
        row["topic_id"]: Topic(
            topic_id=row["topic_id"],
            category=row["category"],
            knowledge_type=row["knowledge_type"],
            information_need=row["information_need"],
        )
        for row in _read_csv(root / "topics.csv")
    }
    questions = {
        row["question_id"]: Question(
            question_id=row["question_id"],
            topic_id=row["topic_id"],
            question_form=row["question_form"],
            language="en",
            text=row["text"],
        )
        for row in _read_csv(root / "questions.csv")
    }
    splits = {row["topic_id"]: row["split"] for row in _read_csv(root / "splits.csv")}
    qrels = load_qrels(root / "qrels.txt")

    collection = Collection(
        sources=sources,
        passages=passages,
        topics=topics,
        questions=questions,
        qrels=qrels,
        splits=splits,
        judgements=load_judgements(root / "judgements.csv"),
        gold=load_gold(root / "gold.csv"),
        pairs=load_pairs(root / "pairs.csv"),
    )
    validate(collection)
    return collection


def validate(collection: Collection) -> None:
    """Raise CollectionError on the first structural problem found."""
    errors: list[str] = []

    for source in collection.sources.values():
        if not source.url.startswith("http"):
            errors.append(f"source {source.source_id}: url must be absolute")
        if not source.access_date:
            errors.append(f"source {source.source_id}: access_date is required")
        if source.risk_category not in RISK_CATEGORIES:
            errors.append(
                f"source {source.source_id}: risk_category must be one of "
                f"{sorted(RISK_CATEGORIES)}"
            )

    for passage in collection.passages.values():
        if passage.source_id not in collection.sources:
            errors.append(f"passage {passage.passage_id}: unknown source {passage.source_id}")
        if passage.topic_id not in collection.topics:
            errors.append(f"passage {passage.passage_id}: unknown topic {passage.topic_id}")
        if not MIN_PASSAGE_WORDS <= passage.word_count <= MAX_PASSAGE_WORDS:
            errors.append(
                f"passage {passage.passage_id}: {passage.word_count} words, expected "
                f"{MIN_PASSAGE_WORDS}-{MAX_PASSAGE_WORDS}"
            )
        if not passage.section_heading:
            errors.append(f"passage {passage.passage_id}: section_heading is required")

    for topic in collection.topics.values():
        if topic.knowledge_type not in KNOWLEDGE_TYPES:
            errors.append(
                f"topic {topic.topic_id}: knowledge_type must be one of "
                f"{sorted(KNOWLEDGE_TYPES)}"
            )
        if topic.topic_id not in collection.splits:
            errors.append(f"topic {topic.topic_id}: not assigned to a split")

    for topic_id, split in collection.splits.items():
        if split not in SPLITS:
            errors.append(f"split for {topic_id}: must be one of {sorted(SPLITS)}")
        if topic_id not in collection.topics:
            errors.append(f"split for {topic_id}: unknown topic")

    for question in collection.questions.values():
        if question.question_form not in QUESTION_FORMS:
            errors.append(
                f"question {question.question_id}: question_form must be one of "
                f"{sorted(QUESTION_FORMS)}"
            )
        if question.language != "en":
            errors.append(f"question {question.question_id}: language must be en")
        judgements = collection.qrels.get(question.question_id, {})
        if question.is_ookb:
            if question.topic_id:
                errors.append(f"question {question.question_id}: ookb must have no topic")
            if judgements:
                errors.append(
                    f"question {question.question_id}: ookb must have no relevance judgements"
                )
        else:
            if question.topic_id not in collection.topics:
                errors.append(
                    f"question {question.question_id}: unknown topic {question.topic_id}"
                )
            if not judgements:
                errors.append(f"question {question.question_id}: no relevance judgements")

    for question_id, judgements in collection.qrels.items():
        if question_id not in collection.questions:
            errors.append(f"qrels: unknown question {question_id}")
        for passage_id in judgements:
            if passage_id not in collection.passages:
                errors.append(f"qrels {question_id}: unknown passage {passage_id}")

    for passage in collection.passages.values():
        if passage.volatility not in VOLATILITY:
            errors.append(
                f"passage {passage.passage_id}: volatility must be one of {sorted(VOLATILITY)}"
            )

    # Each final qrels row must have a matching judgement.
    if collection.judgements:
        judged: set[tuple[str, str]] = set()
        for judgement in collection.judgements:
            if judgement.question_id not in collection.questions:
                errors.append(f"judgements: unknown question {judgement.question_id}")
            if judgement.passage_id not in collection.passages:
                errors.append(f"judgements: unknown passage {judgement.passage_id}")
            if judgement.grade not in (0, 1, 2):
                errors.append(
                    f"judgements {judgement.question_id}/{judgement.passage_id}: "
                    "grade must be 0, 1 or 2"
                )
            if not judgement.judge:
                errors.append("judgements: every row needs a judge")
            judged.add((judgement.question_id, judgement.passage_id))
        for question_id, graded in collection.qrels.items():
            for passage_id in graded:
                if (question_id, passage_id) not in judged:
                    errors.append(
                        f"qrels {question_id}/{passage_id}: no judgement records this grade"
                    )

    for gold in collection.gold.values():
        question = collection.questions.get(gold.question_id)
        if question is None:
            errors.append(f"gold: unknown question {gold.question_id}")
            continue
        if question.is_ookb:
            errors.append(f"gold {gold.question_id}: out-of-KB questions have no gold answer")
            continue
        if not gold.answer.strip():
            errors.append(f"gold {gold.question_id}: answer is empty")
        if not gold.support:
            errors.append(f"gold {gold.question_id}: at least one support sentence is required")
        for passage_id, index in gold.support:
            passage = collection.passages.get(passage_id)
            if passage is None:
                errors.append(f"gold {gold.question_id}: unknown passage {passage_id}")
                continue
            if not 0 <= index < len(passage.sentences):
                errors.append(
                    f"gold {gold.question_id}: {passage_id}#{index} is out of range "
                    f"({len(passage.sentences)} sentences)"
                )
            if collection.qrels.get(gold.question_id, {}).get(passage_id, 0) == 0:
                errors.append(
                    f"gold {gold.question_id}: cites {passage_id}, which the judgements "
                    "do not mark relevant"
                )

    for pair in collection.pairs:
        for topic_id in (pair.topic_a, pair.topic_b):
            if topic_id not in collection.topics:
                errors.append(f"pairs: unknown topic {topic_id}")
        if pair.topic_a == pair.topic_b:
            errors.append(f"pairs: {pair.topic_a} is paired with itself")
        split_a = collection.splits.get(pair.topic_a)
        split_b = collection.splits.get(pair.topic_b)
        if split_a and split_b and split_a != split_b:
            errors.append(
                f"pairs {pair.topic_a}/{pair.topic_b}: a confusable pair must sit in one "
                "split, otherwise the confusion can never be measured"
            )

    if errors:
        raise CollectionError("; ".join(errors[:20]))
