"""Small command-line entry point for the frozen RAG experiment."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import CollectionError, load_collection
from melbourne_mate.evaluation import metrics as metrics_mod
from melbourne_mate.evaluation.collection_quality import (
    build_report,
    containment_bin,
    question_containment,
)
from melbourne_mate.evaluation.generation_runs import write_generation_run
from melbourne_mate.evaluation.runs import read_manifest, read_per_question, write_run
from melbourne_mate.evaluation.stats import compare as compare_runs
from melbourne_mate.evaluation.targeted_qrels import build_targeted_qrels
from melbourne_mate.generation.gate import EvidenceGate
from melbourne_mate.generation.ollama import OllamaClient, OllamaError
from melbourne_mate.pipeline import ARMS, RagPipeline, build_retriever
from melbourne_mate.retrieval.encoders import load_encoder
from melbourne_mate.testing import EchoModel

ENCODER_CHOICES = ["multi-qa-mpnet", "all-minilm", "hashing"]


def _question_ids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        collection = load_collection(args.data)
    except CollectionError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    print(f"  sources: {len(collection.sources)}")
    print(f"  passages: {len(collection.passages)}")
    print(f"  questions: {len(collection.questions)}")
    print(f"  collection fingerprint: {collection.fingerprint()}")
    print(f"  config fingerprint: {CONFIG.fingerprint()}")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    for line in build_report(load_collection(args.data)).lines():
        print(line)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print(CONFIG.to_yaml(), end="")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    collection = load_collection(args.data)
    encoder = load_encoder(args.encoder) if args.arm == "dense" else None
    retriever = build_retriever(collection, args.arm, encoder)
    questions = collection.questions_in_split(args.split)
    if not questions:
        print(f"no questions in split {args.split!r}", file=sys.stderr)
        return 1

    rankings = {
        question.question_id: [
            hit.passage_id
            for hit in retriever.search(question.text, CONFIG.final_top_k)
        ]
        for question in questions
    }
    scores = metrics_mod.score_run(
        rankings,
        collection.qrels,
        [question.question_id for question in questions],
    )
    overlap = question_containment(collection)
    meta = {
        question.question_id: {
            "topic_id": question.topic_id,
            "question_form": question.question_form,
            "knowledge_type": collection.topics[question.topic_id].knowledge_type,
            "category": collection.topics[question.topic_id].category,
            "containment": f"{overlap.get(question.question_id, 0.0):.3f}",
            "containment_bin": containment_bin(overlap.get(question.question_id, 0.0)),
        }
        for question in questions
    }
    path = write_run(
        Path(args.runs_dir) / args.run_id,
        run_id=args.run_id,
        arm=args.arm,
        split=args.split,
        encoder=(
            f"{encoder.model_name}@{getattr(encoder, 'revision', '')}".rstrip("@")
            if encoder
            else "none"
        ),
        collection_fingerprint=collection.fingerprint(),
        rankings=rankings,
        per_question=scores,
        question_meta=meta,
    )
    totals = metrics_mod.aggregate(scores)
    print(f"wrote {path}")
    for metric in metrics_mod.METRICS:
        print(f"  {metric}: {totals[metric]:.4f}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    collection = load_collection(args.data)
    sample_path = Path(args.sample)
    if not sample_path.is_absolute():
        sample_path = Path(args.data) / sample_path
    if not sample_path.exists():
        print(f"sample not found: {sample_path}", file=sys.stderr)
        return 1

    question_ids = _question_ids(sample_path)
    encoder = load_encoder(args.encoder) if args.arm == "dense" else None
    retriever = build_retriever(collection, args.arm, encoder)
    gate = None
    if args.arm != "none":
        gate_encoder = encoder or load_encoder(args.encoder)
        gate = EvidenceGate(gate_encoder)
    model = (
        EchoModel("Offline answer [P001].")
        if args.offline
        else OllamaClient(args.model or None)
    )
    if isinstance(model, OllamaClient):
        try:
            model.resolve_digest()
        except OllamaError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    pipeline = RagPipeline(
        collection,
        retriever,
        model,
        gate=gate,
        arm=args.arm,
    )

    answers = []
    for question_id in question_ids:
        question = collection.questions.get(question_id)
        if question is None:
            print(f"unknown question in sample: {question_id}", file=sys.stderr)
            return 1
        relevant = tuple(
            passage_id
            for passage_id, grade in collection.qrels.get(question_id, {}).items()
            if grade > 0
        )
        answer = pipeline.answer(question.text, relevant_ids=relevant)
        retrieval_hit = any(hit.passage_id in relevant for hit in answer.hits)
        answers.append((question_id, answer, not question.is_ookb, retrieval_hit))

    model_label = "offline" if args.offline else model.model
    path = write_generation_run(
        Path(args.runs_dir) / args.run_id,
        run_id=args.run_id,
        arm=args.arm,
        split=args.split,
        encoder=(
            f"{encoder.model_name}@{getattr(encoder, 'revision', '')}".rstrip("@")
            if encoder
            else "none"
        ),
        collection_fingerprint=collection.fingerprint(),
        answers=answers,
        model=model_label,
    )
    print(f"wrote {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left_manifest = read_manifest(args.left)
    right_manifest = read_manifest(args.right)
    if left_manifest.split != right_manifest.split:
        print("runs use different splits", file=sys.stderr)
        return 1
    if left_manifest.collection_fingerprint != right_manifest.collection_fingerprint:
        print("runs use different collections", file=sys.stderr)
        return 1
    if left_manifest.ndcg_gain != right_manifest.ndcg_gain:
        print("runs use different NDCG gain settings", file=sys.stderr)
        return 1

    collection = load_collection(args.data)
    clusters = {
        question.question_id: question.topic_id
        for question in collection.questions.values()
        if question.topic_id
    }
    result = compare_runs(
        read_per_question(args.left),
        read_per_question(args.right),
        clusters,
        metric=args.metric,
    )
    print(f"mean difference: {result.mean_difference:+.4f}")
    print(f"95% cluster bootstrap CI: [{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    print(f"paired randomisation p: {result.p_value:.4f}")
    return 0


def cmd_targeted_qrels(args: argparse.Namespace) -> int:
    question_ids = _question_ids(args.questions)
    result = build_targeted_qrels(
        args.base_qrels,
        args.verification,
        args.out,
        question_ids=question_ids,
    )
    digest = hashlib.sha256(result.qrels_path.read_bytes()).hexdigest()[:16]
    print(f"wrote {result.qrels_path} (qrels-{digest})")
    print(f"  seed pairs: {result.seed_pairs}")
    print(f"  checked candidates: {result.checked_candidates}")
    print(f"  added positive pairs: {result.added_positive_pairs}")
    return 0


def cmd_evaluate_retrieval(args: argparse.Namespace) -> int:
    """Run the frozen retrieval workflow with one command."""
    bm25_id = f"{args.run_prefix}-bm25-{args.split}"
    dense_label = "mpnet" if args.encoder == "multi-qa-mpnet" else "minilm"
    dense_id = f"{args.run_prefix}-{dense_label}-{args.split}"
    bm25_path = Path(args.runs_dir) / bm25_id
    dense_path = Path(args.runs_dir) / dense_id

    steps = [
        ("validate", cmd_validate, argparse.Namespace(data=args.data)),
        ("quality", cmd_quality, argparse.Namespace(data=args.data)),
        (
            "BM25s retrieval",
            cmd_retrieve,
            argparse.Namespace(
                data=args.data,
                arm="bm25",
                split=args.split,
                run_id=bm25_id,
                encoder=args.encoder,
                runs_dir=args.runs_dir,
            ),
        ),
        (
            "Dense retrieval",
            cmd_retrieve,
            argparse.Namespace(
                data=args.data,
                arm="dense",
                split=args.split,
                run_id=dense_id,
                encoder=args.encoder,
                runs_dir=args.runs_dir,
            ),
        ),
        (
            "comparison",
            cmd_compare,
            argparse.Namespace(
                data=args.data,
                left=bm25_path,
                right=dense_path,
                metric=args.metric,
            ),
        ),
    ]
    for label, command, step_args in steps:
        print(f"\n[{label}]")
        status = command(step_args)
        if status:
            return status
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mm", description="MelbourneMate experiment")
    parser.add_argument("--data", default="data/sample")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.set_defaults(func=cmd_validate)

    quality = sub.add_parser("quality")
    quality.set_defaults(func=cmd_quality)

    config = sub.add_parser("config")
    config.set_defaults(func=cmd_config)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--arm", required=True, choices=["bm25", "dense"])
    retrieve.add_argument(
        "--split", default="validation", choices=["validation", "test"]
    )
    retrieve.add_argument("--run-id", required=True)
    retrieve.add_argument(
        "--encoder",
        default="multi-qa-mpnet",
        choices=ENCODER_CHOICES,
    )
    retrieve.add_argument("--runs-dir", default="runs/retrieval")
    retrieve.set_defaults(func=cmd_retrieve)

    generate = sub.add_parser("generate")
    generate.add_argument("--arm", required=True, choices=ARMS)
    generate.add_argument("--split", default="test", choices=["validation", "test"])
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--sample", required=True)
    generate.add_argument(
        "--encoder",
        default="multi-qa-mpnet",
        choices=ENCODER_CHOICES,
    )
    generate.add_argument("--model", default="")
    generate.add_argument("--offline", action="store_true")
    generate.add_argument("--runs-dir", default="runs/generation")
    generate.set_defaults(func=cmd_generate)

    compare = sub.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--metric", default=CONFIG.evaluation.primary_metric)
    compare.set_defaults(func=cmd_compare)

    qrels = sub.add_parser("targeted-qrels")
    qrels.add_argument("--base-qrels", required=True)
    qrels.add_argument("--verification", required=True)
    qrels.add_argument("--questions", required=True)
    qrels.add_argument("--out", required=True)
    qrels.set_defaults(func=cmd_targeted_qrels)

    workflow = sub.add_parser("evaluate-retrieval")
    workflow.add_argument(
        "--split", default="validation", choices=["validation", "test"]
    )
    workflow.add_argument("--run-prefix", required=True)
    workflow.add_argument(
        "--encoder",
        default="multi-qa-mpnet",
        choices=["multi-qa-mpnet", "all-minilm"],
    )
    workflow.add_argument("--runs-dir", default="runs/retrieval")
    workflow.add_argument("--metric", default=CONFIG.evaluation.primary_metric)
    workflow.set_defaults(func=cmd_evaluate_retrieval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
