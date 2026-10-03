"""Save the remaining design analyses (RQ2, RQ3a and RQ4) from saved runs.

No model is rerun. Retrieval differences come from the saved per-question
metrics and generation outcomes from the saved answer traces.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from melbourne_mate import __version__
from melbourne_mate.config import CONFIG
from melbourne_mate.corpus import Collection, load_qrels
from melbourne_mate.evaluation.generation_runs import read_traces
from melbourne_mate.evaluation.stats import (
    cluster_bootstrap_ci,
    paired_randomisation_test,
    rate_interval,
)

BINS = ("low", "medium", "high")
FORMS = ("canonical", "paraphrased")
ARMS = ("bm25", "mpnet", "no-context")
GROUNDED = ("bm25", "mpnet")
MIN_DISCORDANT = 5  # docs/design.md: fewer discordant questions is insufficient evidence


class RQAnalysisError(ValueError):
    """Raised when the saved runs cannot be compared."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float | None) -> float | str:
    return "" if value is None else round(value, 4)


# ---------- RQ2: retrieval gap by containment and question form ----------


def rq2_rows(
    bm25: Mapping[str, Mapping[str, str]],
    mpnet: Mapping[str, Mapping[str, str]],
    metric: str = "ndcg@5",
) -> list[dict]:
    """MPNet minus BM25s on the saved metric, overall and by slice."""
    if set(bm25) != set(mpnet):
        raise RQAnalysisError("retrieval runs cover different questions")
    groups: list[tuple[str, str, list[str]]] = [("all", "all", sorted(bm25))]
    for dimension, field, values in (
        ("containment_bin", "containment_bin", BINS),
        ("question_form", "question_form", FORMS),
    ):
        for value in values:
            chosen = sorted(q for q, row in bm25.items() if row[field] == value)
            if chosen:
                groups.append((dimension, value, chosen))

    rows = []
    for dimension, value, questions in groups:
        clusters = {q: bm25[q]["topic_id"] for q in questions}
        left = {q: float(bm25[q][metric]) for q in questions}
        right = {q: float(mpnet[q][metric]) for q in questions}
        differences = {q: right[q] - left[q] for q in questions}
        mean, low, high = cluster_bootstrap_ci(differences, clusters)
        rows.append(
            {
                "dimension": dimension,
                "group": value,
                "questions": len(questions),
                "topics": len(set(clusters.values())),
                f"bm25_{metric}": _round(sum(left.values()) / len(left)),
                f"mpnet_{metric}": _round(sum(right.values()) / len(right)),
                "difference": _round(mean),
                "ci_low": _round(low),
                "ci_high": _round(high),
                "p_value": _round(paired_randomisation_test(differences, clusters)),
            }
        )
    return rows


# ---------- RQ3a and RQ4: generation outcomes ----------


def _outcomes(
    traces: Sequence[dict], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, dict]:
    """Per question facts, matching the rules in generation_metrics.py."""
    facts = {}
    for row in traces:
        if row.get("truncated"):
            continue  # truncated answers are reported separately
        relevant = {p for p, grade in qrels.get(row["question_id"], {}).items() if grade > 0}
        facts[row["question_id"]] = {
            "answerable": bool(row["answerable"]),
            "refused": row["status"] == "refused",
            "retrieval_hit": any(p in relevant for p in row.get("retrieved") or []),
        }
    return facts


def _clusters(collection: Collection, questions: Sequence[str]) -> dict[str, str]:
    # Answerable variants share a topic; out-of-KB questions stand alone.
    return {q: collection.questions[q].topic_id or q for q in questions}


def rq3a_rows(collection: Collection, outcomes: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    """Unsupported answers on the same out-of-KB questions, by arm and paired."""
    ookb_sets = [{q for q, f in outcomes[arm].items() if not f["answerable"]} for arm in ARMS]
    if any(s != ookb_sets[0] for s in ookb_sets):
        raise RQAnalysisError("generation runs cover different out-of-KB questions")
    ookb = sorted(ookb_sets[0])
    clusters = _clusters(collection, ookb)
    answered = {arm: {q: not outcomes[arm][q]["refused"] for q in ookb} for arm in ARMS}

    rows = []
    for arm in ARMS:
        rate, low, high = rate_interval(answered[arm], clusters)
        rows.append(
            {
                "comparison": arm,
                "questions": len(ookb),
                "left_rate": "",
                "right_rate": _round(rate),
                "difference": "",
                "ci_low": _round(low),
                "ci_high": _round(high),
                "p_value": "",
                "discordant": "",
                "evidence": "rate",
            }
        )
    for left, right in (("no-context", "bm25"), ("no-context", "mpnet"), ("bm25", "mpnet")):
        differences = {
            q: float(answered[right][q]) - float(answered[left][q]) for q in ookb
        }
        mean, low, high = cluster_bootstrap_ci(differences, clusters)
        discordant = sum(1 for q in ookb if answered[left][q] != answered[right][q])
        sufficient = discordant >= MIN_DISCORDANT
        # docs/design.md: thin comparisons quote no p-value.
        p_value = paired_randomisation_test(differences, clusters) if sufficient else None
        rows.append(
            {
                "comparison": f"{right} minus {left}",
                "questions": len(ookb),
                "left_rate": _round(sum(answered[left].values()) / len(ookb)),
                "right_rate": _round(sum(answered[right].values()) / len(ookb)),
                "difference": _round(mean),
                "ci_low": _round(low),
                "ci_high": _round(high),
                "p_value": _round(p_value),
                "discordant": discordant,
                "evidence": "sufficient" if sufficient else "insufficient",
            }
        )
    return rows


def rq4_rows(collection: Collection, outcomes: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    """Correct and false refusal with intervals, to be read together."""
    rows = []
    for arm in ARMS:
        facts = outcomes[arm]
        ookb = sorted(q for q, f in facts.items() if not f["answerable"])
        # False refusal counts only answerable questions whose evidence was retrieved.
        refusable = sorted(q for q, f in facts.items() if f["answerable"] and f["retrieval_hit"])
        for metric, questions in (("correct_refusal", ookb), ("false_refusal", refusable)):
            if not questions:
                rows.append(
                    {
                        "arm": arm,
                        "metric": metric,
                        "questions": 0,
                        "clusters": 0,
                        "rate": "",
                        "ci_low": "",
                        "ci_high": "",
                        "note": "not applicable without retrieval",
                    }
                )
                continue
            clusters = _clusters(collection, questions)
            refused = {q: facts[q]["refused"] for q in questions}
            rate, low, high = rate_interval(refused, clusters)
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "questions": len(questions),
                    "clusters": len(set(clusters.values())),
                    "rate": _round(rate),
                    "ci_low": _round(low),
                    "ci_high": _round(high),
                    "note": "",
                }
            )
    return rows


def _check_against_runs(rq4: Sequence[dict], summaries: Mapping[str, Mapping[str, str]]) -> None:
    """The saved generation summaries must give the same rates."""
    for row in rq4:
        saved = summaries[row["arm"]].get(f"{row['metric']}_rate", "")
        if row["rate"] == "" or saved == "":
            continue
        if abs(float(saved) - float(row["rate"])) > 1e-3:
            raise RQAnalysisError(
                f"{row['arm']} {row['metric']} {row['rate']} differs from summary.csv {saved}"
            )


# ---------- report ----------


def _summary_rows(rq2: Sequence[dict], rq3a: Sequence[dict], rq4: Sequence[dict]) -> list[dict]:
    by_group = {(r["dimension"], r["group"]): r for r in rq2}
    low = by_group.get(("containment_bin", "low"))
    high = by_group.get(("containment_bin", "high"))
    rows = []
    if low and high:
        gap_grows = low["difference"] > high["difference"] and low["ci_low"] > 0
        rows.append(
            {
                "question": "RQ2",
                "finding": "MPNet gain on low-containment questions",
                "value": low["difference"],
                "ci_low": low["ci_low"],
                "ci_high": low["ci_high"],
                "criterion_met": gap_grows,
                "note": f"high-containment gain {high['difference']} from {high['questions']} questions",
            }
        )
    for row in rq3a:
        if " minus no-context" in row["comparison"]:
            met = (
                row["difference"] < 0
                and row["ci_high"] < 0
                and row["evidence"] == "sufficient"
            )
            rows.append(
                {
                    "question": "RQ3a",
                    "finding": f"Out-of-KB answers, {row['comparison']}",
                    "value": row["difference"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "criterion_met": met,
                    "note": f"{row['discordant']} discordant questions",
                }
            )
        elif row["comparison"] == "mpnet minus bm25":
            rows.append(
                {
                    "question": "RQ3a",
                    "finding": "Out-of-KB answers, MPNet minus BM25s",
                    "value": row["difference"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "criterion_met": row["evidence"] == "sufficient" and row["ci_low"] > 0,
                    "note": f"{row['discordant']} discordant questions, {row['evidence']} evidence",
                }
            )
    for row in rq4:
        if row["rate"] == "":
            continue
        rows.append(
            {
                "question": "RQ4",
                "finding": f"{row['arm']} {row['metric'].replace('_', ' ')}",
                "value": row["rate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "criterion_met": "",
                "note": f"{row['questions']} questions in {row['clusters']} clusters",
            }
        )
    return rows


def _p(value: float | str) -> str:
    return "not reported" if value == "" else f"{float(value):.4f}"


def _pct(value: float | str) -> str:
    return "n/a" if value == "" else f"{float(value):.1%}"


def _report(rq2: Sequence[dict], rq3a: Sequence[dict], rq4: Sequence[dict]) -> str:
    lines = [
        "# Final RQ2, RQ3a and RQ4 analyses",
        "",
        "Computed from saved runs. No model was rerun. Intervals are 95% cluster",
        "bootstrap intervals and p-values come from paired cluster randomisation.",
        "",
        "## RQ2. MPNet minus BM25s on NDCG@5",
        "",
        "| Slice | Questions | Topics | BM25s | MPNet | Difference | 95% CI | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rq2:
        lines.append(
            f"| {r['group']} | {r['questions']} | {r['topics']} | {r['bm25_ndcg@5']:.3f} "
            f"| {r['mpnet_ndcg@5']:.3f} | {r['difference']:+.3f} "
            f"| [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_value']:.4f} |"
        )
    lines += [
        "",
        "Slices are exploratory except the low and high containment comparison.",
        "",
        "## RQ3a. Answers to out-of-KB questions",
        "",
        "| Comparison | Rate before | Rate after | Difference | 95% CI | p | Discordant | Evidence |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for r in rq3a:
        if r["evidence"] == "rate":
            lines.append(
                f"| {r['comparison']} | | {_pct(r['right_rate'])} | | "
                f"[{_pct(r['ci_low'])}, {_pct(r['ci_high'])}] | | | |"
            )
        else:
            lines.append(
                f"| {r['comparison']} | {_pct(r['left_rate'])} | {_pct(r['right_rate'])} "
                f"| {r['difference']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
                f"| {_p(r['p_value'])} | {r['discordant']} | {r['evidence']} |"
            )
    lines += [
        "",
        (
            f"Fewer than {MIN_DISCORDANT} discordant questions is reported as insufficient "
            "evidence, and no p-value is quoted for it."
        ),
        "",
        "## RQ4. Refusal behaviour",
        "",
        "| Arm | Metric | Questions | Rate | 95% CI |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in rq4:
        interval = "" if r["rate"] == "" else f"[{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]"
        lines.append(
            f"| {r['arm']} | {r['metric'].replace('_', ' ')} | {r['questions']} "
            f"| {_pct(r['rate'])} | {interval or r['note']} |"
        )
    lines += [
        "",
        "Correct refusal and false refusal are read together. A system that refuses",
        "everything would score full correct refusal and full false refusal.",
        "False refusal counts only answerable questions whose evidence was retrieved.",
        "",
    ]
    return "\n".join(lines)


def write_rq_analysis(
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
        raise RQAnalysisError(f"analysis directory {out} already exists")

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
    if {m["collection_fingerprint"] for m in manifests.values()} != {collection.fingerprint()}:
        raise RQAnalysisError("runs use a different collection")
    qrels_file = Path(qrels_path)
    qrels_fingerprint = _sha256(qrels_file)
    run_qrels = {m.get("qrels_fingerprint") for m in manifests.values() if m.get("qrels_fingerprint")}
    if run_qrels != {qrels_fingerprint}:
        raise RQAnalysisError("runs use different qrels")

    def metrics(path: Path) -> dict[str, dict[str, str]]:
        return {r["question_id"]: r for r in _read_csv(path / "per_question_metrics.csv")}

    rq2 = rq2_rows(metrics(run_paths["bm25_retrieval"]), metrics(run_paths["mpnet_retrieval"]))

    qrels = load_qrels(qrels_file)
    generation = {
        "bm25": run_paths["bm25_generation"],
        "mpnet": run_paths["mpnet_generation"],
        "no-context": run_paths["no_context_generation"],
    }
    outcomes = {arm: _outcomes(read_traces(path), qrels) for arm, path in generation.items()}
    rq3a = rq3a_rows(collection, outcomes)
    rq4 = rq4_rows(collection, outcomes)
    _check_against_runs(
        rq4,
        {
            arm: {r["metric"]: r["value"] for r in _read_csv(path / "summary.csv")}
            for arm, path in generation.items()
        },
    )

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "rq2-containment.csv", rq2)
    _write_csv(out / "rq3a-grounding.csv", rq3a)
    _write_csv(out / "rq4-refusal.csv", rq4)
    _write_csv(out / "summary.csv", _summary_rows(rq2, rq3a, rq4))
    (out / "report.md").write_text(_report(rq2, rq3a, rq4), encoding="utf-8")

    inputs = {"qrels": qrels_file}
    for name in ("bm25_retrieval", "mpnet_retrieval"):
        inputs[name] = run_paths[name] / "per_question_metrics.csv"
    for name in ("bm25_generation", "mpnet_generation", "no_context_generation"):
        inputs[name] = run_paths[name] / "answers.jsonl"
    manifest = {
        "analysis_id": out.name,
        "code_version": __version__,
        "collection_fingerprint": collection.fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_fingerprints": {name: _sha256(path) for name, path in inputs.items()},
        "input_runs": {name: m["run_id"] for name, m in manifests.items()},
        "min_discordant": MIN_DISCORDANT,
        "permutations": CONFIG.evaluation.randomisation_permutations,
        "qrels_fingerprint": qrels_fingerprint,
        "resamples": CONFIG.evaluation.bootstrap_resamples,
        "seed": CONFIG.evaluation.seed,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
