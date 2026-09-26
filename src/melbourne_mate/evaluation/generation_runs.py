"""Write immutable generation traces, summaries, and run manifests."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from melbourne_mate import __version__
from melbourne_mate.config import CONFIG
from melbourne_mate.evaluation.generation_metrics import (
    GenerationOutcome,
    contingency_2x2,
    summarise_generation,
)
from melbourne_mate.evaluation.runs import RunError
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
    config_fingerprint: str
    questions: int
    created_utc: str
    code_version: str
    python_version: str
    platform: str
    note: str = ""


def trace_row(answer: Answer, question_id: str, answerable: bool) -> dict:
    report = answer.citations
    return {
        "question_id": question_id,
        "question": answer.question,
        "arm": answer.arm,
        "answerable": answerable,
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
    answers: Sequence[tuple[str, Answer, bool, bool]],
    model: str,
    note: str = "",
) -> Path:
    """Write a generation run.

    answers: (question_id, Answer, answerable, retrieval_hit) in question order.
    """
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise RunError(f"run directory {path} already exists; choose a new run id")
    path.mkdir(parents=True, exist_ok=True)

    outcomes: list[GenerationOutcome] = []
    with (path / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for question_id, answer, answerable, retrieval_hit in answers:
            handle.write(
                json.dumps(trace_row(answer, question_id, answerable), ensure_ascii=False) + "\n"
            )
            outcomes.append(
                GenerationOutcome(
                    question_id=question_id,
                    answerable=answerable,
                    refused=answer.refused,
                    citations_valid=bool(answer.citations and answer.citations.is_valid),
                    retrieval_hit=retrieval_hit,
                    # Keep truncation separate from answers and refusals.
                    truncated=answer.truncated,
                )
            )

    summary = summarise_generation(outcomes).as_dict()
    if arm == "none":
        # No retrieved evidence, so these rates are undefined.
        summary["false_refusal_rate"] = None
        summary["citation_validity_rate"] = None
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            writer.writerow([metric, round(value, 6) if isinstance(value, float) else value])

    table = contingency_2x2(outcomes)
    with (path / "contingency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
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
        config_fingerprint=CONFIG.fingerprint(),
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
    path = Path(directory) / "answers.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
