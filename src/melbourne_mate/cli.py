"""Small command-line entry point for the frozen RAG experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import CollectionError, load_collection, load_qrels
from melbourne_mate.evaluation import metrics as metrics_mod
from melbourne_mate.evaluation.answer_review import (
    AnswerReviewError,
    prepare_answer_review,
    score_answer_review,
)
from melbourne_mate.evaluation.collection_quality import (
    build_report,
    containment_bin,
    question_containment,
)
from melbourne_mate.evaluation.error_analysis import (
    ErrorAnalysisError,
    write_error_analysis,
)
from melbourne_mate.evaluation.generation_inputs import (
    GenerationCase,
    GenerationInputError,
    generation_sample_fingerprint,
    load_generation_cases,
    load_generation_qrels,
)
from melbourne_mate.evaluation.generation_runs import (
    GenerationRunItem,
    write_generation_run,
)
from melbourne_mate.evaluation.runs import (
    read_manifest,
    read_per_question,
    read_run_file,
    write_run,
)
from melbourne_mate.evaluation.stats import compare as compare_runs
from melbourne_mate.evaluation.study import StudyError, summarise_study
from melbourne_mate.evaluation.targeted_qrels import (
    build_review_pool,
    build_targeted_qrels,
)
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
    """Validate the collection and print its reproducibility fingerprints."""

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
    """Print the collection quality checks used before an experiment."""

    for line in build_report(load_collection(args.data)).lines():
        print(line)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show the frozen experiment configuration."""

    print(CONFIG.to_yaml(), end="")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Run one retriever and save its rankings and evaluation scores."""

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
    """Run one generation arm and save its evidence."""
    collection = load_collection(args.data)
    sample_path = Path(args.sample)
    if not sample_path.is_absolute():
        sample_path = Path(args.data) / sample_path
    if not sample_path.exists():
        print(f"sample not found: {sample_path}", file=sys.stderr)
        return 1

    qrels_path = getattr(args, "qrels", None)
    try:
        if qrels_path:
            cases = load_generation_cases(sample_path, collection)
            qrels, qrels_fingerprint = load_generation_qrels(
                qrels_path, collection
            )
        else:
            cases = tuple(
                GenerationCase(question_id, "general")
                for question_id in _question_ids(sample_path)
            )
            qrels = collection.qrels
            qrels_fingerprint = ""
    except (GenerationInputError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sample_fingerprint = generation_sample_fingerprint(cases)
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

    answers: list[GenerationRunItem] = []
    for case in cases:
        question = collection.questions.get(case.question_id)
        if question is None:
            print(f"unknown question in sample: {case.question_id}", file=sys.stderr)
            return 1
        relevant = tuple(
            passage_id
            for passage_id, grade in qrels.get(case.question_id, {}).items()
            if grade > 0
        )
        answer = pipeline.answer(question.text, relevant_ids=relevant)
        retrieval_hit = any(hit.passage_id in relevant for hit in answer.hits)
        answers.append(
            GenerationRunItem(
                question_id=case.question_id,
                answer=answer,
                answerable=not question.is_ookb,
                retrieval_hit=retrieval_hit,
                risk_category=case.risk_category,
                language=question.language,
                question_form=question.question_form,
            )
        )

    model_label = "offline" if args.offline else model.model
    model_digest = "" if args.offline else model.digest
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
        model_digest=model_digest,
        sample_fingerprint=sample_fingerprint,
        qrels_fingerprint=qrels_fingerprint,
    )
    print(f"wrote {path}")
    return 0


def cmd_evaluate_generation(args: argparse.Namespace) -> int:
    """Run all three generation arms with one checked input set."""
    sample_path = Path(args.sample)
    if not sample_path.is_absolute():
        sample_path = Path(args.data) / sample_path
    try:
        collection = load_collection(args.data)
        load_generation_cases(sample_path, collection)
        load_generation_qrels(args.qrels, collection)
    except (CollectionError, GenerationInputError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_ids = {
        "bm25": f"{args.run_prefix}-bm25",
        "dense": f"{args.run_prefix}-mpnet",
        "none": f"{args.run_prefix}-no-context",
    }
    for run_id in run_ids.values():
        target = Path(args.runs_dir) / run_id
        if target.exists():
            print(f"target run already exists: {target}", file=sys.stderr)
            return 1

    for arm in ("bm25", "dense", "none"):
        status = cmd_generate(
            argparse.Namespace(
                data=args.data,
                arm=arm,
                split="test",
                run_id=run_ids[arm],
                # cmd_generate owns path resolution.
                sample=args.sample,
                qrels=args.qrels,
                encoder=args.encoder,
                model=args.model,
                offline=args.offline,
                runs_dir=args.runs_dir,
            )
        )
        if status:
            return status
    return 0


def cmd_analyse_errors(args: argparse.Namespace) -> int:
    """Create one report from saved retrieval and generation runs."""
    try:
        path = write_error_analysis(
            args.out,
            load_collection(args.data),
            args.qrels,
            args.bm25_retrieval,
            args.mpnet_retrieval,
            args.bm25_generation,
            args.mpnet_generation,
            args.no_context_generation,
        )
    except (CollectionError, ErrorAnalysisError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def cmd_prepare_answer_review(args: argparse.Namespace) -> int:
    """Create the fixed answer-review sheets from saved runs."""
    try:
        path = prepare_answer_review(
            args.out,
            load_collection(args.data),
            {
                "bm25": args.bm25_run,
                "mpnet": args.mpnet_run,
                "no-context": args.no_context_run,
            },
            size=args.size,
        )
    except (AnswerReviewError, CollectionError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def cmd_score_answer_review(args: argparse.Namespace) -> int:
    """Summarise two completed answer-review sheets."""
    try:
        path = score_answer_review(args.review_dir)
    except (AnswerReviewError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def cmd_study_summary(args: argparse.Namespace) -> int:
    """Summarise the student study responses."""
    try:
        path = summarise_study(args.study_dir)
    except (StudyError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {path / 'summary.csv'} and {path / 'summary.md'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two compatible saved runs with topic-level statistics."""

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
    # Different qrels produce incomparable scores.
    if left_manifest.qrels_fingerprint != right_manifest.qrels_fingerprint:
        print("runs use different qrels", file=sys.stderr)
        return 1

    collection = load_collection(args.data)
    if collection.fingerprint() != left_manifest.collection_fingerprint:
        print("data uses a different collection from the runs", file=sys.stderr)
        return 1
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
    if getattr(args, "out", None):
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "metric": args.metric,
            "left_run": left_manifest.run_id,
            "right_run": right_manifest.run_id,
            "mean_difference": f"{result.mean_difference:.6f}",
            "ci_low": f"{result.ci_low:.6f}",
            "ci_high": f"{result.ci_high:.6f}",
            "p_value": f"{result.p_value:.6f}",
            "clusters": result.clusters,
            "observations": result.observations,
            "resamples": result.resamples,
            "permutations": CONFIG.evaluation.randomisation_permutations,
            "seed": result.seed,
            "qrels_fingerprint": left_manifest.qrels_fingerprint,
        }
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(row), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow(row)
        print(f"wrote {output}")
    return 0


def cmd_rescore_retrieval(args: argparse.Namespace) -> int:
    """Recalculate metrics from a saved ranking without running retrieval."""
    collection = load_collection(args.data)
    source = Path(args.source_run)
    manifest = read_manifest(source)
    if manifest.collection_fingerprint != collection.fingerprint():
        print("source run uses a different collection", file=sys.stderr)
        return 1
    if manifest.config_fingerprint != CONFIG.fingerprint():
        print("source run uses a different configuration", file=sys.stderr)
        return 1

    # Keep rankings fixed and rescore only.
    rankings = read_run_file(source)
    questions = collection.questions_in_split(manifest.split)
    question_ids = [question.question_id for question in questions]
    if set(rankings) != set(question_ids):
        print("source run does not match its recorded split", file=sys.stderr)
        return 1

    qrels_path = Path(args.qrels)
    qrels = load_qrels(qrels_path)
    try:
        scores = metrics_mod.score_run(rankings, qrels, question_ids)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    overlap = question_containment(collection)
    meta = {
        question.question_id: {
            "topic_id": question.topic_id,
            "question_form": question.question_form,
            "knowledge_type": collection.topics[question.topic_id].knowledge_type,
            "category": collection.topics[question.topic_id].category,
            "containment": f"{overlap.get(question.question_id, 0.0):.3f}",
            "containment_bin": containment_bin(
                overlap.get(question.question_id, 0.0)
            ),
        }
        for question in questions
    }
    qrels_fp = hashlib.sha256(qrels_path.read_bytes()).hexdigest()[:16]
    path = write_run(
        Path(args.runs_dir) / args.run_id,
        run_id=args.run_id,
        arm=manifest.arm,
        split=manifest.split,
        encoder=manifest.encoder,
        collection_fingerprint=manifest.collection_fingerprint,
        rankings=rankings,
        per_question=scores,
        question_meta=meta,
        note=f"Rescored from {manifest.run_id} using qrels-{qrels_fp}",
        qrels_fingerprint=qrels_fp,
    )
    print(f"wrote {path}")
    for metric, value in metrics_mod.aggregate(scores).items():
        print(f"  {metric}: {value:.4f}")
    return 0


def cmd_targeted_qrels(args: argparse.Namespace) -> int:
    """Apply checked candidate grades to the seed qrels."""

    question_ids = (
        sorted(read_run_file(args.run)) if args.run else _question_ids(args.questions)
    )
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


def cmd_qrels_pool(args: argparse.Namespace) -> int:
    """Create the small MPNet candidate sheet for targeted review."""

    result = build_review_pool(load_collection(args.data), args.run, args.out)
    print(f"wrote {result.path}")
    print(f"  candidates: {result.candidates}")
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
    """Register the ``mm`` commands and dispatch the selected workflow."""

    parser = argparse.ArgumentParser(prog="mm", description="MelbourneMate experiment")
    parser.add_argument("--data", default="data/sample")
    sub = parser.add_subparsers(dest="command", required=True)

    # Collection checks run before model experiments.
    validate = sub.add_parser("validate")
    validate.set_defaults(func=cmd_validate)

    quality = sub.add_parser("quality")
    quality.set_defaults(func=cmd_quality)

    config = sub.add_parser("config")
    config.set_defaults(func=cmd_config)

    # Run and save one retrieval method.
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

    # Run one generation arm or the full three-arm comparison.
    generate = sub.add_parser("generate")
    generate.add_argument("--arm", required=True, choices=ARMS)
    generate.add_argument("--split", default="test", choices=["validation", "test"])
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--sample", required=True)
    generate.add_argument("--qrels")
    generate.add_argument(
        "--encoder",
        default="multi-qa-mpnet",
        choices=ENCODER_CHOICES,
    )
    generate.add_argument("--model", default="")
    generate.add_argument("--offline", action="store_true")
    generate.add_argument("--runs-dir", default="runs/generation")
    generate.set_defaults(func=cmd_generate)

    generation_workflow = sub.add_parser("evaluate-generation")
    generation_workflow.add_argument("--run-prefix", required=True)
    generation_workflow.add_argument("--sample", required=True)
    generation_workflow.add_argument("--qrels", required=True)
    generation_workflow.add_argument(
        "--encoder", default="multi-qa-mpnet", choices=ENCODER_CHOICES
    )
    generation_workflow.add_argument("--model", default="")
    generation_workflow.add_argument("--offline", action="store_true")
    generation_workflow.add_argument("--runs-dir", default="runs/generation")
    generation_workflow.set_defaults(func=cmd_evaluate_generation)

    # Analyse, compare or rescore results already saved on disk.
    analysis = sub.add_parser("analyse-errors")
    analysis.add_argument("--qrels", required=True)
    analysis.add_argument("--bm25-retrieval", required=True)
    analysis.add_argument("--mpnet-retrieval", required=True)
    analysis.add_argument("--bm25-generation", required=True)
    analysis.add_argument("--mpnet-generation", required=True)
    analysis.add_argument("--no-context-generation", required=True)
    analysis.add_argument("--out", required=True)
    analysis.set_defaults(func=cmd_analyse_errors)

    prepare_review = sub.add_parser("prepare-answer-review")
    prepare_review.add_argument("--bm25-run", required=True)
    prepare_review.add_argument("--mpnet-run", required=True)
    prepare_review.add_argument("--no-context-run", required=True)
    prepare_review.add_argument("--size", type=int, default=30)
    prepare_review.add_argument("--out", required=True)
    prepare_review.set_defaults(func=cmd_prepare_answer_review)

    score_review = sub.add_parser("score-answer-review")
    score_review.add_argument("--review-dir", required=True)
    score_review.set_defaults(func=cmd_score_answer_review)

    study = sub.add_parser("study-summary")
    study.add_argument("--study-dir", default="study")
    study.set_defaults(func=cmd_study_summary)

    compare = sub.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--metric", default=CONFIG.evaluation.primary_metric)
    compare.add_argument("--out")
    compare.set_defaults(func=cmd_compare)

    rescore = sub.add_parser("rescore-retrieval")
    rescore.add_argument("--source-run", required=True)
    rescore.add_argument("--qrels", required=True)
    rescore.add_argument("--run-id", required=True)
    rescore.add_argument("--runs-dir", default="runs/retrieval")
    rescore.set_defaults(func=cmd_rescore_retrieval)

    # Keep manual qrels work limited to new MPNet candidates.
    qrels = sub.add_parser("targeted-qrels")
    qrels.add_argument("--base-qrels", required=True)
    qrels.add_argument("--verification", required=True)
    question_source = qrels.add_mutually_exclusive_group(required=True)
    question_source.add_argument("--run")
    question_source.add_argument("--questions")
    qrels.add_argument("--out", required=True)
    qrels.set_defaults(func=cmd_targeted_qrels)

    pool = sub.add_parser("qrels-pool")
    pool.add_argument("--run", required=True)
    pool.add_argument("--out", required=True)
    pool.set_defaults(func=cmd_qrels_pool)

    # Run the complete frozen retrieval workflow in the required order.
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
