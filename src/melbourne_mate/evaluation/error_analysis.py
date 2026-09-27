"""Build error cases from saved retrieval and generation runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from melbourne_mate import __version__
from melbourne_mate.corpus import RISK_CATEGORIES, Collection, load_qrels
from melbourne_mate.evaluation.generation_runs import read_traces
from melbourne_mate.evaluation.risk import HIGH_RISK_CATEGORIES


class ErrorAnalysisError(ValueError):
    """Raised when saved runs cannot be compared safely."""


@dataclass(frozen=True)
class ErrorCase:
    question_id: str
    stage: str
    arm: str
    error_type: str
    risk_category: str
    high_risk: bool
    question_form: str
    bm25_ndcg5: float | None = None
    mpnet_ndcg5: float | None = None
    retrieval_hit: bool | None = None
    status: str = ""
    citation_valid: bool | None = None


def _index_traces(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in indexed:
            raise ErrorAnalysisError(f"duplicate trace for {question_id}")
        indexed[question_id] = row
    return indexed


def _case(
    question_id: str,
    stage: str,
    arm: str,
    error_type: str,
    trace: Mapping[str, object],
    **values: object,
) -> ErrorCase:
    risk = str(trace["risk_category"])
    if risk not in RISK_CATEGORIES:
        raise ErrorAnalysisError(f"unknown risk category {risk}")
    return ErrorCase(
        question_id=question_id,
        stage=stage,
        arm=arm,
        error_type=error_type,
        risk_category=risk,
        # Keep every report on the shared risk list.
        high_risk=risk in HIGH_RISK_CATEGORIES,
        question_form=str(trace["question_form"]),
        **values,
    )


def build_error_cases(
    bm25_metrics: Mapping[str, Mapping[str, object]],
    mpnet_metrics: Mapping[str, Mapping[str, object]],
    generation: Mapping[str, Sequence[Mapping[str, object]]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[ErrorCase, ...]:
    """Classify failures from fixed rankings and saved answers."""
    bm25_ids = set(bm25_metrics)
    mpnet_ids = set(mpnet_metrics)
    if bm25_ids != mpnet_ids:
        raise ErrorAnalysisError("retrieval runs contain different questions")

    required_arms = {"bm25", "mpnet", "no-context"}
    if set(generation) != required_arms:
        raise ErrorAnalysisError("generation runs must include all three arms")
    traces = {arm: _index_traces(generation[arm]) for arm in required_arms}
    generation_ids = {arm: set(rows) for arm, rows in traces.items()}
    if len({frozenset(ids) for ids in generation_ids.values()}) != 1:
        raise ErrorAnalysisError("generation runs contain different questions")
    if not bm25_ids.issubset(generation_ids["bm25"]):
        raise ErrorAnalysisError("retrieval questions are missing from generation runs")

    cases: list[ErrorCase] = []
    for question_id in sorted(bm25_ids):
        bm25 = bm25_metrics[question_id]
        mpnet = mpnet_metrics[question_id]
        trace = traces["bm25"][question_id]
        bm25_ndcg = float(bm25["ndcg@5"])
        mpnet_ndcg = float(mpnet["ndcg@5"])
        common = {"bm25_ndcg5": bm25_ndcg, "mpnet_ndcg5": mpnet_ndcg}

        if bm25_ndcg > mpnet_ndcg:
            cases.append(
                _case(question_id, "retrieval", "bm25", "bm25_better", trace, **common)
            )
        if mpnet_ndcg > bm25_ndcg and trace["question_form"] == "paraphrased":
            cases.append(
                _case(
                    question_id,
                    "retrieval",
                    "mpnet",
                    "mpnet_better_paraphrase",
                    trace,
                    **common,
                )
            )
        if float(bm25["recall@5"]) == 0 and float(mpnet["recall@5"]) == 0:
            cases.append(
                _case(
                    question_id,
                    "retrieval",
                    "both",
                    "both_retrieval_fail",
                    trace,
                    **common,
                )
            )

    for arm in ("bm25", "mpnet", "no-context"):
        for question_id, trace in sorted(traces[arm].items()):
            answerable = bool(trace["answerable"])
            status = str(trace["status"])
            citations = trace.get("citations")
            citation_valid = (
                bool(citations.get("valid"))
                if isinstance(citations, Mapping) and arm != "no-context"
                else None
            )
            relevant = {
                passage_id
                for passage_id, grade in qrels.get(question_id, {}).items()
                if grade > 0
            }
            # No-context has no retrieval stage.
            retrieval_hit = (
                None
                if arm == "no-context"
                else any(item in relevant for item in trace["retrieved"])
            )
            values = {
                "retrieval_hit": retrieval_hit,
                "status": status,
                "citation_valid": citation_valid,
            }

            if bool(trace.get("truncated")):
                cases.append(
                    _case(question_id, "generation", arm, "truncated", trace, **values)
                )
                continue
            if answerable and retrieval_hit and status == "refused":
                cases.append(
                    _case(
                        question_id,
                        "generation",
                        arm,
                        "retrieval_hit_but_refused",
                        trace,
                        **values,
                    )
                )
            if (
                answerable
                and arm != "no-context"
                and retrieval_hit is False
                and status == "answered"
            ):
                cases.append(
                    _case(
                        question_id,
                        "generation",
                        arm,
                        "retrieval_miss_but_answered",
                        trace,
                        **values,
                    )
                )
            if (
                answerable
                and arm != "no-context"
                and status == "answered"
                and not citation_valid
            ):
                cases.append(
                    _case(
                        question_id,
                        "generation",
                        arm,
                        "invalid_citation",
                        trace,
                        **values,
                    )
                )
            if not answerable and status == "answered":
                cases.append(
                    _case(
                        question_id, "generation", arm, "ookb_answered", trace, **values
                    )
                )

    return tuple(
        sorted(
            cases,
            key=lambda item: (item.stage, item.error_type, item.arm, item.question_id),
        )
    )


def _load_retrieval_metrics(directory: str | Path) -> dict[str, dict[str, object]]:
    path = Path(directory) / "per_question_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["question_id"]: row for row in csv.DictReader(handle)}


def _summary_rows(cases: Sequence[ErrorCase]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics = (
        ("retrieval", "bm25", "bm25_better"),
        ("retrieval", "mpnet", "mpnet_better_paraphrase"),
        ("retrieval", "both", "both_retrieval_fail"),
        ("generation", "bm25", "retrieval_hit_but_refused"),
        ("generation", "mpnet", "retrieval_hit_but_refused"),
        ("generation", "bm25", "retrieval_miss_but_answered"),
        ("generation", "mpnet", "retrieval_miss_but_answered"),
        ("generation", "bm25", "invalid_citation"),
        ("generation", "mpnet", "invalid_citation"),
        ("generation", "bm25", "ookb_answered"),
        ("generation", "mpnet", "ookb_answered"),
        ("generation", "no-context", "ookb_answered"),
    )
    for stage, arm, metric in metrics:
        rows.append(
            {
                "stage": stage,
                "arm": arm,
                "metric": metric,
                "value": sum(
                    case.stage == stage
                    and case.arm == arm
                    and case.error_type == metric
                    for case in cases
                ),
            }
        )
    for arm in ("bm25", "mpnet", "no-context"):
        high_risk = {
            case.question_id
            for case in cases
            if case.stage == "generation" and case.arm == arm and case.high_risk
        }
        rows.append(
            {
                "stage": "generation",
                "arm": arm,
                "metric": "high_risk_failure",
                "value": len(high_risk),
            }
        )
    return rows


def _report(cases: Sequence[ErrorCase], summary: Sequence[Mapping[str, object]]) -> str:
    values = {
        (str(row["arm"]), str(row["metric"])): int(row["value"]) for row in summary
    }

    def examples(error_type: str, arm: str | None = None) -> str:
        selected = [
            f"{case.question_id} ({case.arm}, {case.risk_category})"
            for case in cases
            if case.error_type == error_type and (arm is None or case.arm == arm)
        ]
        return ", ".join(selected[:3]) or "None"

    def questions(count: int) -> str:
        return "question" if count == 1 else "questions"

    bm25_better = values[("bm25", "bm25_better")]
    mpnet_better = values[("mpnet", "mpnet_better_paraphrase")]
    both_fail = values[("both", "both_retrieval_fail")]

    lines = [
        "# Sprint 9 error analysis",
        "",
        "This report uses saved rankings and answers. No model was rerun.",
        "",
        "## Retrieval",
        "",
        f"- BM25s has higher NDCG@5 on {bm25_better} {questions(bm25_better)}.",
        (
            "- MPNet has higher NDCG@5 on "
            f"{mpnet_better} paraphrased {questions(mpnet_better)}."
        ),
        (
            "- Both retrievers miss all relevant top-five passages on "
            f"{both_fail} {questions(both_fail)}."
        ),
        "",
        "## Generation",
        "",
        "| Arm | Hit but refused | Miss but answered | Invalid citation | OOKB answered | High-risk failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("bm25", "mpnet", "no-context"):
        lines.append(
            "| "
            f"{arm} | {values.get((arm, 'retrieval_hit_but_refused'), 0)} | "
            f"{values.get((arm, 'retrieval_miss_but_answered'), 0)} | "
            f"{values.get((arm, 'invalid_citation'), 0)} | "
            f"{values.get((arm, 'ookb_answered'), 0)} | "
            f"{values[(arm, 'high_risk_failure')]} |"
        )
    lines.extend(
        [
            "",
            "## Example question IDs",
            "",
            f"- BM25s better: {examples('bm25_better')}",
            f"- MPNet better paraphrases: {examples('mpnet_better_paraphrase')}",
            f"- Both retrieval methods fail: {examples('both_retrieval_fail')}",
            f"- Retrieval hit but refused: {examples('retrieval_hit_but_refused')}",
            f"- Retrieval miss but answered: {examples('retrieval_miss_but_answered')}",
            f"- Invalid citation: {examples('invalid_citation')}",
            f"- OOKB answered: {examples('ookb_answered')}",
            "",
            "The CSV files contain every case. These counts are descriptive and do not replace human answer review.",
            "",
        ]
    )
    return "\n".join(lines)


def write_error_analysis(
    directory: str | Path,
    collection: Collection,
    qrels_path: str | Path,
    bm25_retrieval: str | Path,
    mpnet_retrieval: str | Path,
    bm25_generation: str | Path,
    mpnet_generation: str | Path,
    no_context_generation: str | Path,
) -> Path:
    """Validate saved runs and write one immutable analysis folder."""
    out = Path(directory)
    if out.exists() and any(out.iterdir()):
        raise ErrorAnalysisError(f"analysis directory {out} already exists")

    run_paths = {
        "bm25_retrieval": Path(bm25_retrieval),
        "mpnet_retrieval": Path(mpnet_retrieval),
        "bm25_generation": Path(bm25_generation),
        "mpnet_generation": Path(mpnet_generation),
        "no_context_generation": Path(no_context_generation),
    }
    manifests = {
        name: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for name, path in run_paths.items()
    }
    collection_fingerprints = {
        manifest["collection_fingerprint"] for manifest in manifests.values()
    }
    if collection_fingerprints != {collection.fingerprint()}:
        raise ErrorAnalysisError("runs use different collections")

    qrels_file = Path(qrels_path)
    qrels_fingerprint = hashlib.sha256(qrels_file.read_bytes()).hexdigest()[:16]
    run_qrels = {
        manifest.get("qrels_fingerprint")
        for manifest in manifests.values()
        if manifest.get("qrels_fingerprint")
    }
    if run_qrels != {qrels_fingerprint}:
        raise ErrorAnalysisError("runs use different qrels")

    generation = {
        "bm25": read_traces(bm25_generation),
        "mpnet": read_traces(mpnet_generation),
        "no-context": read_traces(no_context_generation),
    }
    cases = build_error_cases(
        _load_retrieval_metrics(bm25_retrieval),
        _load_retrieval_metrics(mpnet_retrieval),
        generation,
        load_qrels(qrels_file),
    )
    summary = _summary_rows(cases)

    out.mkdir(parents=True, exist_ok=True)
    case_columns = (
        list(asdict(cases[0])) if cases else list(ErrorCase.__dataclass_fields__)
    )
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=case_columns, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow(asdict(case))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "arm", "metric", "value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)
    (out / "report.md").write_text(_report(cases, summary), encoding="utf-8")

    manifest = {
        "analysis_id": out.name,
        "case_count": len(cases),
        "code_version": __version__,
        "collection_fingerprint": collection.fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "high_risk_categories": sorted(HIGH_RISK_CATEGORIES),
        "input_runs": {
            name: manifest["run_id"] for name, manifest in manifests.items()
        },
        "qrels_fingerprint": qrels_fingerprint,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
