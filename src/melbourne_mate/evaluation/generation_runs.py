"""Write immutable generation traces, summaries, and run manifests."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from melbourne_mate import __version__
from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import Collection
from melbourne_mate.evaluation.generation_metrics import (
    GenerationOutcome,
    contingency_2x2,
    summarise_generation,
)
from melbourne_mate.evaluation.runs import RunError
from melbourne_mate.generation.citations import check_citations
from melbourne_mate.pipeline import Answer


@dataclass(frozen=True)
class GenerationManifest:
    run_id: str
    arm: str
    split: str
    provider: str
    model: str
    seed: int
    temperature: float
    prompt_version: str
    encoder: str
    collection_fingerprint: str
    sample_fingerprint: str
    qrels_fingerprint: str
    config_fingerprint: str
    model_digest: str
    questions: int
    created_utc: str
    code_version: str
    python_version: str
    platform: str
    note: str = ""


@dataclass(frozen=True)
class GenerationRunItem:
    question_id: str
    answer: Answer
    answerable: bool
    retrieval_hit: bool
    risk_category: str = "general"
    language: str = "en"
    question_form: str = "canonical"


def trace_row(item: GenerationRunItem) -> dict:
    """Flatten one answer into a saved trace."""
    answer = item.answer
    report = answer.citations
    return {
        "question_id": item.question_id,
        "question": answer.question,
        "arm": answer.arm,
        "answerable": item.answerable,
        "risk_category": item.risk_category,
        "language": item.language,
        "question_form": item.question_form,
        "status": answer.status,
        "truncated": answer.truncated,
        "answer": answer.text,
        "retrieved": [hit.passage_id for hit in answer.hits],
        "gate": None
        if answer.gate is None
        else {
            "sufficient": answer.gate.sufficient,
            "top_score": round(answer.gate.top_score, 4),
            "supporting": answer.gate.supporting,
            "reason": answer.gate.reason,
        },
        "citations": None
        if report is None
        else {
            "cited": list(report.cited),
            "unsupported": list(report.unsupported),
            "irrelevant": list(report.irrelevant),
            "uncited_urls": list(report.uncited_urls),
            "valid": report.is_valid,
        },
        "latency_ms": answer.latency_ms,
        "model": answer.meta.get("model", ""),
    }


def write_generation_run(
    directory: str | Path,
    run_id: str,
    arm: str,
    split: str,
    encoder: str,
    collection_fingerprint: str,
    answers: Sequence[GenerationRunItem],
    model: str,
    sample_fingerprint: str = "",
    qrels_fingerprint: str = "",
    model_digest: str = "",
    note: str = "",
) -> Path:
    """Write one immutable generation run."""
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise RunError(f"run directory {path} already exists; choose a new run id")
    path.mkdir(parents=True, exist_ok=True)

    outcomes: list[GenerationOutcome] = []
    with (path / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for item in answers:
            handle.write(
                json.dumps(trace_row(item), ensure_ascii=False) + "\n"
            )
            outcomes.append(
                GenerationOutcome(
                    question_id=item.question_id,
                    answerable=item.answerable,
                    refused=item.answer.refused,
                    citations_valid=bool(
                        item.answer.citations and item.answer.citations.is_valid
                    ),
                    retrieval_hit=item.retrieval_hit,
                    # Track truncation on its own.
                    truncated=item.answer.truncated,
                    risk_category=item.risk_category,
                    language=item.language,
                    question_form=item.question_form,
                )
            )

    summary = summarise_generation(outcomes).as_dict()
    if arm == "none":
        # No-context has no retrieval evidence.
        summary["false_refusal_rate"] = None
        summary["citation_validity_rate"] = None
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            writer.writerow([metric, round(value, 6) if isinstance(value, float) else value])

    table = contingency_2x2(outcomes)
    with (path / "contingency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell", "count", "reads_as"])
        writer.writerow(["hit_answered", table["hit_answered"], "working as intended"])
        writer.writerow(["hit_refused", table["hit_refused"], "generator too conservative"])
        writer.writerow(["miss_answered", table["miss_answered"], "answered without evidence"])
        writer.writerow(["miss_refused", table["miss_refused"], "retrieval is the bottleneck"])

    manifest = GenerationManifest(
        run_id=run_id,
        arm=arm,
        split=split,
        provider=CONFIG.generation.provider,
        model=model,
        seed=CONFIG.generation.seed,
        temperature=CONFIG.generation.temperature,
        prompt_version=CONFIG.generation.prompt_version,
        encoder=encoder,
        collection_fingerprint=collection_fingerprint,
        sample_fingerprint=sample_fingerprint,
        qrels_fingerprint=qrels_fingerprint,
        config_fingerprint=CONFIG.fingerprint(),
        model_digest=model_digest,
        questions=len(answers),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        note=note,
    )
    (path / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_traces(directory: str | Path) -> list[dict]:
    """Read saved answer traces."""
    path = Path(directory) / "answers.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rescore_generation_run(
    directory: str | Path,
    *,
    source_run: str | Path,
    collection: Collection,
    qrels: Mapping[str, Mapping[str, int]],
    qrels_fingerprint: str,
    run_id: str,
) -> Path:
    """Recheck a saved run after qrels change, without rerunning the model."""
    output = Path(directory)
    if output.exists() and any(output.iterdir()):
        raise RunError(f"run directory {output} already exists; choose a new run id")

    source = Path(source_run)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["collection_fingerprint"] != collection.fingerprint():
        raise RunError("saved run does not match this collection")

    traces = read_traces(source)
    rescored: list[dict] = []
    outcomes: list[GenerationOutcome] = []
    # Reuse saved model output; only evidence-dependent labels may change.
    for trace in traces:
        question_id = trace["question_id"]
        retrieved = tuple(trace.get("retrieved", ()))
        relevant = tuple(qrels.get(question_id, {}))
        citation_data = trace.get("citations")
        if citation_data is not None:
            # Rebuild citation inputs from the passages retrieved in the original run.
            source_urls = {
                passage_id: collection.sources[
                    collection.passages[passage_id].source_id
                ].url
                for passage_id in retrieved
                if passage_id in collection.passages
            }
            passage_texts = {
                passage_id: collection.passages[passage_id].text
                for passage_id in retrieved
                if passage_id in collection.passages
            }
            report = check_citations(
                trace["answer"],
                retrieved_ids=retrieved,
                relevant_ids=relevant,
                source_urls=source_urls,
                passage_texts=passage_texts,
            )
            citation_data = {
                "cited": list(report.cited),
                "unsupported": list(report.unsupported),
                "irrelevant": list(report.irrelevant),
                "uncited_urls": list(report.uncited_urls),
                "valid": report.is_valid,
            }

        updated = dict(trace)
        updated["citations"] = citation_data
        rescored.append(updated)
        outcomes.append(
            GenerationOutcome(
                question_id=question_id,
                answerable=bool(trace["answerable"]),
                refused=trace["status"] == "refused",
                citations_valid=bool(citation_data and citation_data["valid"]),
                retrieval_hit=any(passage_id in relevant for passage_id in retrieved),
                truncated=bool(trace.get("truncated", False)),
                risk_category=trace.get("risk_category", "general"),
                language=trace.get("language", "en"),
                question_form=trace.get("question_form", "canonical"),
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for trace in rescored:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

    summary = summarise_generation(outcomes).as_dict()
    if manifest["arm"] == "none":
        # These metrics require retrieved evidence, so they are undefined here.
        summary["false_refusal_rate"] = None
        summary["citation_validity_rate"] = None
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            writer.writerow(
                [metric, round(value, 6) if isinstance(value, float) else value]
            )

    table = contingency_2x2(outcomes)
    with (output / "contingency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell", "count", "reads_as"])
        writer.writerow(["hit_answered", table["hit_answered"], "working as intended"])
        writer.writerow(
            ["hit_refused", table["hit_refused"], "generator too conservative"]
        )
        writer.writerow(
            ["miss_answered", table["miss_answered"], "answered without evidence"]
        )
        writer.writerow(
            ["miss_refused", table["miss_refused"], "retrieval is the bottleneck"]
        )

    manifest["run_id"] = run_id
    manifest["qrels_fingerprint"] = qrels_fingerprint
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["code_version"] = __version__
    manifest["python_version"] = sys.version.split()[0]
    manifest["platform"] = platform.platform()
    manifest["note"] = (
        f"Rescored from {source.name} after a qrels correction; "
        "model output unchanged."
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
