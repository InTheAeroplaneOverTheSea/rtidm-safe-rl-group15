#!/usr/bin/env python3
"""Validate an author-v1 bridge and write one plain-text comparison report.

The input collection must contain a matched ``nominal`` scenario and one or
more stress scenarios produced by the author-v1 stress harness.  The script
supports both the one-episode ("1-seed") gate and ten-episode ("10-seed")
confirmation.  It never submits jobs or queries Slurm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aggregate_author_v1_dual_outcomes import (
    DualOutcomeError,
    MetricResult,
    aggregate_dual_outcomes,
)


ALGORITHMS = (
    "CPO",
    "CUP",
    "FOCOPS",
    "OnCRPO",
    "PCPO",
    "PPOLag",
    "RCPO",
    "TRPOLag",
)
STRESS_SCENARIOS = (
    "gaussian_sigma15",
    "bias_positive",
    "bias_negative",
    "random_walk_drift",
    "dropout_markov_v1",
    "unannounced_meal_60g",
)
METRICS = ("TIR", "Risk", "CV")
OUTCOME_SOURCES = (
    ("true_bg", "true physiological BG"),
    ("sensor_cgm", "exposed sensor CGM"),
)
METRIC_HEADERS = {
    "TIR (%)": "TIR",
    "Risk index": "Risk",
    "CV (%)": "CV",
}
FINAL_RE = re.compile(
    r"^final mean=([-+0-9.eE]+), sample SD=([-+0-9.eE]+),"
)
EARLY_RE = re.compile(r"^Episodes shorter than 2016 steps: (\d+)/(\d+)$")


class SummaryError(RuntimeError):
    """Raised when a collection is incomplete or fails validation."""


@dataclass(frozen=True)
class ArmResult:
    metrics_by_source: dict[str, dict[str, MetricResult]]
    early_terminated: int
    total_episodes: int
    completeness: dict[str, int]
    delivered_unannounced_meals: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a matched author-v1 nominal/stress collection and write "
            "one Google-Docs-friendly plain-text report."
        )
    )
    parser.add_argument(
        "result_collection",
        type=Path,
        help="Collection directory containing scenario/replicateN directories.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        choices=(1, 10),
        help="Episodes (literal evaluation seeds) per cell: 1 or 10.",
    )
    parser.add_argument("--replicate-id", type=int, default=0)
    parser.add_argument("--logit-penalty", type=float, default=10.0)
    parser.add_argument("--base-evaluation-seed", type=int, default=22)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        help=(
            "Algorithms to summarize. By default, discover every algorithm "
            "present in nominal/no_shield."
        ),
    )
    parser.add_argument(
        "--stress-scenarios",
        nargs="+",
        choices=STRESS_SCENARIOS,
        help=(
            "Stress scenarios to compare with nominal. By default, discover "
            "all supported non-nominal scenario directories in the collection."
        ),
    )
    parser.add_argument(
        "--expected-result-collection",
        help=(
            "Metadata result_collection value. Defaults to the input "
            "directory name."
        ),
    )
    parser.add_argument(
        "--paper-baseline-collection",
        type=Path,
        help=(
            "Optional dual-trace original-paper baseline collection. When set, "
            "include its no-Shield and predictive results before the matched "
            "stress-harness scenarios."
        ),
    )
    parser.add_argument(
        "--expected-paper-baseline-collection",
        help="Metadata name for --paper-baseline-collection (defaults to its directory name).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output text file. Defaults to validation_reports/<collection>_"
            "r<replicate>_<episodes>seed_summary.txt."
        ),
    )
    return parser.parse_args()


def number_tag(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def discover_algorithms(collection: Path, replicate_id: int) -> list[str]:
    root = collection / "nominal" / f"replicate{replicate_id}" / "no_shield" / "t1d"
    if not root.is_dir():
        raise SummaryError(f"missing matched nominal result directory: {root}")
    present = {path.name for path in root.iterdir() if path.is_dir()}
    algorithms = [algorithm for algorithm in ALGORITHMS if algorithm in present]
    if not algorithms:
        raise SummaryError(f"no supported algorithms found under {root}")
    return algorithms


def discover_stress_scenarios(collection: Path) -> list[str]:
    present = {path.name for path in collection.iterdir() if path.is_dir()}
    scenarios = [scenario for scenario in STRESS_SCENARIOS if scenario in present]
    if not scenarios:
        raise SummaryError(
            "no supported stress scenario found; the collection must contain "
            "nominal and at least one non-nominal scenario"
        )
    return scenarios


def run_validator(command: list[str], label: str) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        details = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        raise SummaryError(f"{label} failed\n{details}")
    return completed.stdout


def parse_aggregation(output: str, label: str) -> tuple[dict[str, MetricResult], int, int]:
    if "Validation: PASS" not in output.splitlines():
        raise SummaryError(f"{label} did not print Validation: PASS")

    current_metric: str | None = None
    metrics: dict[str, MetricResult] = {}
    early: tuple[int, int] | None = None
    for line in output.splitlines():
        if line in METRIC_HEADERS:
            current_metric = METRIC_HEADERS[line]
            continue
        early_match = EARLY_RE.match(line)
        if early_match:
            early = (int(early_match.group(1)), int(early_match.group(2)))
            continue
        final_match = FINAL_RE.match(line)
        if final_match and current_metric is not None:
            metrics[current_metric] = MetricResult(
                mean=float(final_match.group(1)),
                sample_sd=float(final_match.group(2)),
            )
            current_metric = None

    missing = [metric for metric in METRICS if metric not in metrics]
    if missing:
        raise SummaryError(f"{label} is missing metrics: {', '.join(missing)}")
    if early is None:
        raise SummaryError(f"{label} is missing the early-termination count")
    return metrics, early[0], early[1]


def count_files(root: Path) -> dict[str, int]:
    patterns = {
        "summaries": "summary_results.csv",
        "details": "detailed_results.csv",
        "run metadata": "run_metadata.txt",
        "stress metadata": "stress_episode_metadata.json",
    }
    return {
        label: sum(1 for _ in root.rglob(filename))
        for label, filename in patterns.items()
    }


def count_delivered_unannounced_meals(root: Path) -> int:
    delivered = 0
    for path in root.rglob("stress_episode_metadata.json"):
        try:
            payload = json.loads(path.read_text())
            episodes = payload["episodes"]
            delivered += sum(
                len(episode["delivered_unannounced_meals"])
                for episode in episodes
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SummaryError(f"cannot read delivered meals from {path}: {exc}") from exc
    return delivered


def read_job_ids(
    collection: Path,
    scenario: str,
    replicate_id: int,
    expected_episodes: int,
) -> dict[tuple[str, str, str], list[str]]:
    record = collection / scenario / f"replicate{replicate_id}" / "submitted_jobs.tsv"
    if not record.is_file():
        return {}

    jobs: dict[tuple[str, str, str], list[str]] = {}
    with record.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                row_replicate = int(row["replicate_id"])
                row_episodes = int(row["episodes"])
                key = (row["algorithm"], row["condition"], row["phase"])
                job_id = row["job_id"]
            except (KeyError, TypeError, ValueError) as exc:
                raise SummaryError(f"invalid submission record {record}: {exc}") from exc
            if row["stress_scenario"] != scenario:
                raise SummaryError(f"{record}: scenario mismatch in submission row")
            if row_replicate != replicate_id:
                raise SummaryError(f"{record}: replicate mismatch in submission row")
            if row_episodes != expected_episodes:
                raise SummaryError(f"{record}: episode-count mismatch in submission row")
            if not job_id.isdigit():
                raise SummaryError(f"{record}: invalid Slurm job id {job_id!r}")
            jobs.setdefault(key, []).append(job_id)
    return jobs


def read_paper_baseline_job_ids(
    collection: Path,
    expected_episodes: int,
) -> dict[tuple[str, str, str], list[str]]:
    record = collection / "submitted_jobs.tsv"
    if not record.is_file():
        return {}

    jobs: dict[tuple[str, str, str], list[str]] = {}
    with record.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                row_episodes = int(row["episodes"])
                key = (row["algorithm"], row["condition"], row["phase"])
                job_id = row["job_id"]
            except (KeyError, TypeError, ValueError) as exc:
                raise SummaryError(f"invalid submission record {record}: {exc}") from exc
            if row.get("experiment") != "paper_baseline":
                raise SummaryError(f"{record}: expected paper_baseline submission rows")
            if row.get("replicate_id") != "0":
                raise SummaryError(f"{record}: paper baseline replicate must be 0")
            if row_episodes != expected_episodes:
                raise SummaryError(f"{record}: episode-count mismatch in submission row")
            if not job_id.isdigit():
                raise SummaryError(f"{record}: invalid Slurm job id {job_id!r}")
            jobs.setdefault(key, []).append(job_id)
    return jobs


def format_job_ids(
    jobs: dict[tuple[str, str, str], list[str]],
    algorithm: str,
    condition: str,
    phase: str,
) -> str:
    values = jobs.get((algorithm, condition, phase), [])
    return ", ".join(values) if values else "N/A (submission record unavailable)"


def validate_arm(
    *,
    project_dir: Path,
    collection: Path,
    expected_collection: str,
    scenario: str | None,
    replicate_id: int,
    episodes: int,
    base_evaluation_seed: int,
    logit_penalty: float,
    algorithm: str,
    condition: str,
) -> tuple[ArmResult, Path]:
    penalty_tag = number_tag(logit_penalty)
    stress_harness = scenario is not None
    scenario_root = (
        collection / scenario / f"replicate{replicate_id}"
        if stress_harness
        else collection
    )
    if condition == "no_shield":
        root = scenario_root / "no_shield" / "t1d" / algorithm
    else:
        root = scenario_root / f"predictive_penalty{penalty_tag}" / "t1d" / algorithm
    if not root.is_dir():
        raise SummaryError(f"missing result directory: {root}")

    completeness = count_files(root)
    required_counts = ("summaries", "details", "run metadata")
    if stress_harness:
        required_counts += ("stress metadata",)
    incomplete = {
        name: completeness[name]
        for name in required_counts
        if completeness[name] != 81
    }
    if incomplete:
        formatted = ", ".join(f"{name}={count}" for name, count in incomplete.items())
        scenario_label = scenario or "paper_baseline"
        raise SummaryError(
            f"{algorithm} {scenario_label} {condition} is incomplete: {formatted}"
        )

    command = [
        sys.executable,
        str(project_dir / "scripts" / "aggregate_author_v1_t1d.py"),
        str(root),
        "--expected-episodes",
        str(episodes),
        "--algorithm",
        algorithm,
        "--condition",
        condition,
        "--expected-result-collection",
        expected_collection,
    ]
    if stress_harness:
        command.extend(
            (
                "--stress-scenario",
                scenario,
                "--replicate-id",
                str(replicate_id),
                "--base-evaluation-seed",
                str(base_evaluation_seed),
            )
        )
    if condition == "predictive":
        command.extend(("--logit-penalty", str(logit_penalty)))
    scenario_label = scenario or "paper_baseline"
    output = run_validator(
        command, f"{algorithm} {scenario_label} {condition} aggregation"
    )
    _, early_terminated, total_episodes = parse_aggregation(
        output, f"{algorithm} {scenario_label} {condition} aggregation"
    )
    try:
        dual = aggregate_dual_outcomes(
            root,
            expected_episodes=episodes,
            condition=condition,
            logit_penalty=logit_penalty,
            canonical_source="true_bg" if stress_harness else "sensor_cgm",
            require_baseline_dual_metadata=not stress_harness,
        )
    except DualOutcomeError as exc:
        raise SummaryError(
            f"{algorithm} {scenario_label} {condition} dual-outcome aggregation failed: {exc}"
        ) from exc
    delivered = (
        count_delivered_unannounced_meals(root)
        if scenario == "unannounced_meal_60g"
        else None
    )
    return (
        ArmResult(
            metrics_by_source=dual.metrics,
            early_terminated=early_terminated,
            total_episodes=total_episodes,
            completeness=completeness,
            delivered_unannounced_meals=delivered,
        ),
        root,
    )


def validate_pair(
    *,
    project_dir: Path,
    no_shield_root: Path,
    predictive_root: Path,
    algorithm: str,
    scenario: str,
    episodes: int,
    replicate_id: int,
    base_evaluation_seed: int,
    logit_penalty: float,
) -> None:
    command = [
        sys.executable,
        str(project_dir / "scripts" / "validate_author_v1_stress_pairing.py"),
        str(no_shield_root),
        str(predictive_root),
        "--expected-episodes",
        str(episodes),
        "--expected-cells",
        "81",
        "--replicate-id",
        str(replicate_id),
        "--base-evaluation-seed",
        str(base_evaluation_seed),
        "--logit-penalty",
        str(logit_penalty),
    ]
    output = run_validator(command, f"{algorithm} {scenario} pairing validation")
    if "Pairing validation: PASS" not in output.splitlines():
        raise SummaryError(
            f"{algorithm} {scenario} pairing validator did not print PASS"
        )


def format_metric(result: MetricResult) -> str:
    return f"{result.mean:.6f} ± {result.sample_sd:.6f}"


def format_delta(value: float) -> str:
    return f"{value:+.6f}"


def append_outcome_results(
    lines: list[str],
    *,
    source: str,
    source_label: str,
    no_shield: ArmResult,
    predictive: ArmResult,
    nominal: dict[str, ArmResult] | None,
) -> None:
    no_metrics = no_shield.metrics_by_source[source]
    predictive_metrics = predictive.metrics_by_source[source]
    lines.extend((f"Outcome source: {source_label}", "", "Results (mean +/- sample SD):", "", "- no Shield:"))
    for metric in METRICS:
        lines.append(f"  - {metric} = {format_metric(no_metrics[metric])}")
    lines.append("- predictive:")
    for metric in METRICS:
        lines.append(f"  - {metric} = {format_metric(predictive_metrics[metric])}")
    lines.append("- Shield-minus-no-Shield:")
    shield_deltas: dict[str, float] = {}
    for metric in METRICS:
        delta = predictive_metrics[metric].mean - no_metrics[metric].mean
        shield_deltas[metric] = delta
        lines.append(f"  - {metric} = {format_delta(delta)}")

    lines.extend(("", "Matched nominal comparison:", ""))
    if nominal is None:
        lines.extend(
            (
                "- no-Shield stress-minus-nominal: N/A",
                "- predictive stress-minus-nominal: N/A",
            )
        )
    else:
        for condition_label, current, matched in (
            ("no-Shield", no_shield, nominal["no_shield"]),
            ("predictive", predictive, nominal["predictive"]),
        ):
            lines.append(f"- {condition_label} stress-minus-nominal:")
            current_metrics = current.metrics_by_source[source]
            matched_metrics = matched.metrics_by_source[source]
            for metric in METRICS:
                delta = current_metrics[metric].mean - matched_metrics[metric].mean
                lines.append(f"  - {metric} = {format_delta(delta)}")

    lines.extend(("", "Potential Shield break indicators:", ""))
    indicators = []
    if shield_deltas["TIR"] < 0:
        indicators.append(f"TIR worsened by {format_delta(shield_deltas['TIR'])}.")
    if shield_deltas["Risk"] > 0:
        indicators.append(f"Risk worsened by {format_delta(shield_deltas['Risk'])}.")
    if shield_deltas["CV"] > 0:
        indicators.append(f"CV worsened by {format_delta(shield_deltas['CV'])}.")
    if indicators:
        lines.extend(f"- {indicator}" for indicator in indicators)
    else:
        lines.append("-")
    lines.append("")


def append_scenario_report(
    lines: list[str],
    *,
    algorithm: str,
    scenario: str,
    replicate_id: int,
    episodes: int,
    logit_penalty: float,
    jobs: dict[tuple[str, str, str], list[str]],
    no_shield: ArmResult,
    predictive: ArmResult,
    nominal: dict[str, ArmResult] | None,
    pairing_status: str = "PASS",
    stress_harness: bool = True,
) -> None:
    lines.extend(
        (
            f"Scenario: {scenario}",
            "",
            f"Replicate id: {replicate_id}",
            f"Episodes per cell: {episodes}",
            f"Logit penalty: {number_tag(logit_penalty)}",
            "",
            "Slurm jobs:",
            "",
            "- no-Shield smoke: "
            + format_job_ids(jobs, algorithm, "no_shield", "smoke"),
            "- no-Shield full: "
            + format_job_ids(jobs, algorithm, "no_shield", "full"),
            "- predictive smoke: "
            + format_job_ids(jobs, algorithm, "predictive", "smoke"),
            "- predictive full: "
            + format_job_ids(jobs, algorithm, "predictive", "full"),
            "",
            "Completeness:",
            "",
            "- no Shield: "
            f"{no_shield.completeness['summaries']} summaries / "
            f"{no_shield.completeness['details']} details / "
            f"{no_shield.completeness['run metadata']} run metadata / "
            + (
                f"{no_shield.completeness['stress metadata']} stress metadata"
                if stress_harness
                else "N/A stress metadata"
            ),
            "- predictive: "
            f"{predictive.completeness['summaries']} summaries / "
            f"{predictive.completeness['details']} details / "
            f"{predictive.completeness['run metadata']} run metadata / "
            + (
                f"{predictive.completeness['stress metadata']} stress metadata"
                if stress_harness
                else "N/A stress metadata"
            ),
            "",
            "Validation:",
            "",
            "- no-Shield aggregation: PASS",
            "- predictive aggregation: PASS",
            f"- paired disturbances: {pairing_status}",
            "",
        )
    )
    for source, source_label in OUTCOME_SOURCES:
        append_outcome_results(
            lines,
            source=source,
            source_label=source_label,
            no_shield=no_shield,
            predictive=predictive,
            nominal=nominal,
        )

    lines.extend(
        (
            "",
            "Notes / unresolved discrepancies:",
            "",
            "- Episodes shorter than 2016 steps: "
            f"no Shield {no_shield.early_terminated}/{no_shield.total_episodes}; "
            f"predictive {predictive.early_terminated}/{predictive.total_episodes}.",
        )
    )
    if no_shield.delivered_unannounced_meals is not None:
        lines.append(
            "- Delivered unannounced meals: "
            f"no Shield {no_shield.delivered_unannounced_meals}; "
            f"predictive {predictive.delivered_unannounced_meals}."
        )
    if stress_harness:
        lines.extend(("- All completeness, aggregation, and pairing checks pass.", ""))
    else:
        lines.extend(("- Both aggregations and dual-trace checks pass.", ""))


def append_paper_bridge_diagnostic(
    lines: list[str],
    *,
    paper: dict[str, ArmResult],
    bridge_nominal: dict[str, ArmResult],
) -> None:
    lines.extend(
        (
            "Paper-evaluator versus bridge-nominal diagnostic:",
            "",
            "This is a harness-change diagnostic, not the stress-effect denominator.",
            "",
        )
    )
    for source, source_label in OUTCOME_SOURCES:
        lines.extend((f"Outcome source: {source_label}", ""))
        for condition_label, condition in (
            ("no-Shield", "no_shield"),
            ("predictive", "predictive"),
        ):
            lines.append(f"- {condition_label} bridge-nominal-minus-paper-baseline:")
            bridge_metrics = bridge_nominal[condition].metrics_by_source[source]
            paper_metrics = paper[condition].metrics_by_source[source]
            for metric in METRICS:
                delta = bridge_metrics[metric].mean - paper_metrics[metric].mean
                lines.append(f"  - {metric} = {format_delta(delta)}")
        lines.append("")


def build_report(
    *,
    expected_collection: str,
    algorithms: list[str],
    stress_scenarios: list[str],
    replicate_id: int,
    episodes: int,
    base_evaluation_seed: int,
    logit_penalty: float,
    results: dict[tuple[str, str, str], ArmResult],
    jobs_by_scenario: dict[str, dict[tuple[str, str, str], list[str]]],
    paper_baseline_collection: str | None = None,
    paper_baseline_results: dict[tuple[str, str], ArmResult] | None = None,
    paper_baseline_jobs: dict[tuple[str, str, str], list[str]] | None = None,
) -> str:
    first_seed = base_evaluation_seed + replicate_id * 10_000
    last_seed = first_seed + episodes - 1
    seed_text = str(first_seed) if first_seed == last_seed else f"{first_seed}..{last_seed}"
    lines = [
        "Author-v1 validation + stress summary",
        "",
        f"Algorithms: {', '.join(algorithms)}",
        f"Result collection: {expected_collection}",
        f"Replicate id: {replicate_id}",
        f"Episodes per cell: {episodes}",
        f"Evaluation/simulator seeds: {seed_text}",
        f"Logit penalty: {number_tag(logit_penalty)}",
        "Outcome sources: true physiological BG and exposed sensor CGM",
        "Sensor outcome definition: the CGM trace supplied to the policy and Shield",
        "",
        "Matched comparisons use nominal from this same stress-harness collection.",
        "Do not use an older validation collection as the stress-minus-nominal denominator.",
        "",
    ]

    scenarios = ["nominal", *stress_scenarios]
    for algorithm in algorithms:
        lines.extend((f"Algorithm: {algorithm}", ""))
        if paper_baseline_collection is not None:
            if paper_baseline_results is None or paper_baseline_jobs is None:
                raise SummaryError("paper baseline report inputs are incomplete")
            lines.extend(
                (
                    "Experiment: original paper evaluator reproduction",
                    f"Result collection: {paper_baseline_collection}",
                    "Canonical paper metric source: exposed sensor CGM",
                    "",
                )
            )
            append_scenario_report(
                lines,
                algorithm=algorithm,
                scenario="paper_baseline_nominal",
                replicate_id=0,
                episodes=episodes,
                logit_penalty=logit_penalty,
                jobs=paper_baseline_jobs,
                no_shield=paper_baseline_results[(algorithm, "no_shield")],
                predictive=paper_baseline_results[(algorithm, "predictive")],
                nominal=None,
                pairing_status="N/A (original paper evaluator)",
                stress_harness=False,
            )
        lines.extend(
            (
                "Experiment: matched stress-harness validation bridge",
                f"Result collection: {expected_collection}",
                "",
            )
        )
        nominal = {
            condition: results[(algorithm, "nominal", condition)]
            for condition in ("no_shield", "predictive")
        }
        for scenario in scenarios:
            append_scenario_report(
                lines,
                algorithm=algorithm,
                scenario=scenario,
                replicate_id=replicate_id,
                episodes=episodes,
                logit_penalty=logit_penalty,
                jobs=jobs_by_scenario[scenario],
                no_shield=results[(algorithm, scenario, "no_shield")],
                predictive=results[(algorithm, scenario, "predictive")],
                nominal=None if scenario == "nominal" else nominal,
            )
        if paper_baseline_collection is not None:
            assert paper_baseline_results is not None
            append_paper_bridge_diagnostic(
                lines,
                paper={
                    condition: paper_baseline_results[(algorithm, condition)]
                    for condition in ("no_shield", "predictive")
                },
                bridge_nominal=nominal,
            )
        if lines[-1] != "":
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.replicate_id < 0:
        raise SystemExit("--replicate-id must be nonnegative")
    if args.logit_penalty < 0:
        raise SystemExit("--logit-penalty must be nonnegative")
    if args.base_evaluation_seed < 0:
        raise SystemExit("--base-evaluation-seed must be nonnegative")

    try:
        project_dir = Path(__file__).resolve().parent.parent
        collection = args.result_collection.resolve()
        if not collection.is_dir():
            raise SummaryError(f"result collection does not exist: {collection}")
        expected_collection = args.expected_result_collection or collection.name
        algorithms = args.algorithms or discover_algorithms(
            collection, args.replicate_id
        )
        stress_scenarios = args.stress_scenarios or discover_stress_scenarios(
            collection
        )
        output = args.output
        if output is None:
            output = (
                project_dir
                / "validation_reports"
                / (
                    f"{expected_collection}_r{args.replicate_id}_"
                    f"{args.episodes}seed_summary.txt"
                )
            )
        else:
            output = output.resolve()

        scenarios = ["nominal", *stress_scenarios]
        results: dict[tuple[str, str, str], ArmResult] = {}
        jobs_by_scenario = {
            scenario: read_job_ids(
                collection, scenario, args.replicate_id, args.episodes
            )
            for scenario in scenarios
        }
        paper_collection: Path | None = None
        expected_paper_collection: str | None = None
        paper_results: dict[tuple[str, str], ArmResult] | None = None
        paper_jobs: dict[tuple[str, str, str], list[str]] | None = None
        if args.paper_baseline_collection is not None:
            paper_collection = args.paper_baseline_collection.resolve()
            if not paper_collection.is_dir():
                raise SummaryError(
                    f"paper baseline collection does not exist: {paper_collection}"
                )
            expected_paper_collection = (
                args.expected_paper_baseline_collection or paper_collection.name
            )
            paper_jobs = read_paper_baseline_job_ids(
                paper_collection, args.episodes
            )
            paper_results = {}
            for algorithm in algorithms:
                for condition in ("no_shield", "predictive"):
                    arm, _ = validate_arm(
                        project_dir=project_dir,
                        collection=paper_collection,
                        expected_collection=expected_paper_collection,
                        scenario=None,
                        replicate_id=0,
                        episodes=args.episodes,
                        base_evaluation_seed=args.base_evaluation_seed,
                        logit_penalty=args.logit_penalty,
                        algorithm=algorithm,
                        condition=condition,
                    )
                    paper_results[(algorithm, condition)] = arm

        for algorithm in algorithms:
            for scenario in scenarios:
                no_shield, no_shield_root = validate_arm(
                    project_dir=project_dir,
                    collection=collection,
                    expected_collection=expected_collection,
                    scenario=scenario,
                    replicate_id=args.replicate_id,
                    episodes=args.episodes,
                    base_evaluation_seed=args.base_evaluation_seed,
                    logit_penalty=args.logit_penalty,
                    algorithm=algorithm,
                    condition="no_shield",
                )
                predictive, predictive_root = validate_arm(
                    project_dir=project_dir,
                    collection=collection,
                    expected_collection=expected_collection,
                    scenario=scenario,
                    replicate_id=args.replicate_id,
                    episodes=args.episodes,
                    base_evaluation_seed=args.base_evaluation_seed,
                    logit_penalty=args.logit_penalty,
                    algorithm=algorithm,
                    condition="predictive",
                )
                validate_pair(
                    project_dir=project_dir,
                    no_shield_root=no_shield_root,
                    predictive_root=predictive_root,
                    algorithm=algorithm,
                    scenario=scenario,
                    episodes=args.episodes,
                    replicate_id=args.replicate_id,
                    base_evaluation_seed=args.base_evaluation_seed,
                    logit_penalty=args.logit_penalty,
                )
                results[(algorithm, scenario, "no_shield")] = no_shield
                results[(algorithm, scenario, "predictive")] = predictive

        report = build_report(
            expected_collection=expected_collection,
            algorithms=algorithms,
            stress_scenarios=stress_scenarios,
            replicate_id=args.replicate_id,
            episodes=args.episodes,
            base_evaluation_seed=args.base_evaluation_seed,
            logit_penalty=args.logit_penalty,
            results=results,
            jobs_by_scenario=jobs_by_scenario,
            paper_baseline_collection=expected_paper_collection,
            paper_baseline_results=paper_results,
            paper_baseline_jobs=paper_jobs,
        )
        write_atomic(output, report)
    except SummaryError as exc:
        print(f"Summary generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Summary written: {output}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Episodes per cell: {args.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
