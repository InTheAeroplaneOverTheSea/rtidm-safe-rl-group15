#!/usr/bin/env python3
"""Aggregate true-BG and exposed-CGM outcomes from controller traces.

This module deliberately uses only the Python standard library.  It computes
episode metrics from the raw ``BG`` and ``CGM`` columns, then follows the same
aggregation hierarchy as ``aggregate_author_v1_t1d.py``:

    episode -> patient -> cohort -> policy-training seed -> final mean/SD

``CGM`` is the exposed sensor trace supplied to the policy and Shield.  The
pre-gap ``Native_CGM`` column remains a diagnostic and is not an outcome source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


COHORTS = ("child", "adolescent", "adult")
PATIENT_NUMBERS = tuple(range(2, 11))
POLICY_SEEDS = (0, 1, 2)
OUTCOME_COLUMNS = {
    "true_bg": "BG",
    "sensor_cgm": "CGM",
}
METRICS = ("TIR", "Risk", "CV")


class DualOutcomeError(RuntimeError):
    """Raised when raw traces cannot support a valid dual-outcome report."""


@dataclass(frozen=True)
class MetricResult:
    mean: float
    sample_sd: float


@dataclass(frozen=True)
class DualOutcomeResult:
    metrics: dict[str, dict[str, MetricResult]]
    episode_rows: int


def number_tag(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def glucose_metrics(values: list[float]) -> dict[str, float]:
    if not values:
        raise DualOutcomeError("cannot calculate metrics from an empty trace")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise DualOutcomeError("glucose trace contains a non-finite or nonpositive value")

    count = len(values)
    mean = statistics.fmean(values)
    tir = 100.0 * sum(70.0 <= value <= 180.0 for value in values) / count
    risk = statistics.fmean(
        10.0 * (1.509 * (math.log(value) ** 1.084 - 5.381)) ** 2
        for value in values
    )
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    cv = 100.0 * math.sqrt(variance) / (mean + 1e-6)
    return {"TIR": tir, "Risk": risk, "CV": cv}


def read_detailed_rows(path: Path, expected_episodes: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise DualOutcomeError(f"missing {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        raise DualOutcomeError(
            f"{path}: expected {expected_episodes} episodes, found {len(rows)}"
        )
    expected_ids = list(range(expected_episodes))
    try:
        actual_ids = [int(row["episode"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise DualOutcomeError(f"{path}: invalid episode column ({exc})") from exc
    if actual_ids != expected_ids:
        raise DualOutcomeError(
            f"{path}: expected episode IDs {expected_ids}, found {actual_ids}"
        )
    return rows


def read_trace(path: Path, expected_length: int) -> dict[str, list[float]]:
    if not path.is_file():
        raise DualOutcomeError(f"missing {path}")
    traces = {source: [] for source in OUTCOME_COLUMNS}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(OUTCOME_COLUMNS.values()) - fieldnames
        if missing:
            raise DualOutcomeError(
                f"{path}: missing outcome columns {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            for source, column in OUTCOME_COLUMNS.items():
                try:
                    traces[source].append(float(row[column]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise DualOutcomeError(
                        f"{path}:{row_number}: invalid {column} value"
                    ) from exc
    actual_length = len(traces["true_bg"])
    if actual_length != expected_length:
        raise DualOutcomeError(
            f"{path}: expected {expected_length} controller rows, found {actual_length}"
        )
    return traces


def compare_canonical_metrics(
    *,
    detailed_row: dict[str, str],
    computed: dict[str, float],
    path: Path,
    tolerance: float,
) -> None:
    columns = {
        "TIR": "time_in_range_pct",
        "Risk": "risk_index",
        "CV": "cv_pct",
    }
    for metric, column in columns.items():
        try:
            recorded = float(detailed_row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise DualOutcomeError(f"{path}: invalid canonical {column}") from exc
        if not math.isclose(recorded, computed[metric], rel_tol=0.0, abs_tol=tolerance):
            raise DualOutcomeError(
                f"{path}: recomputed {metric}={computed[metric]:.9f} does not "
                f"match recorded {recorded:.9f} within {tolerance}"
            )


def aggregate_dual_outcomes(
    root: Path,
    *,
    expected_episodes: int,
    condition: str,
    logit_penalty: float = 10.0,
    canonical_source: str,
    require_baseline_dual_metadata: bool = False,
    canonical_tolerance: float = 1e-3,
) -> DualOutcomeResult:
    if expected_episodes < 1:
        raise DualOutcomeError("expected_episodes must be positive")
    if condition not in ("no_shield", "predictive"):
        raise DualOutcomeError(f"unsupported condition {condition!r}")
    if canonical_source not in OUTCOME_COLUMNS:
        raise DualOutcomeError(f"unsupported canonical source {canonical_source!r}")
    root = root.resolve()
    result_tag = (
        "no_shield"
        if condition == "no_shield"
        else f"with_shield_{number_tag(logit_penalty)}"
    )

    cell_episode_metrics: dict[
        tuple[str, int, int], dict[str, list[dict[str, float]]]
    ] = {}
    episode_rows = 0
    for cohort in COHORTS:
        for patient_number in PATIENT_NUMBERS:
            patient = f"{cohort}#{patient_number:03d}"
            for policy_seed in POLICY_SEEDS:
                leaf = root / patient / f"seed{policy_seed}" / result_tag
                metadata_path = leaf / "run_metadata.txt"
                if not metadata_path.is_file():
                    raise DualOutcomeError(f"missing {metadata_path}")
                metadata = read_key_values(metadata_path)
                if require_baseline_dual_metadata:
                    expected_metadata = {
                        "dual_outcome_trace": "true",
                        "canonical_outcome_source": "exposed_sensor_cgm",
                    }
                    for key, expected in expected_metadata.items():
                        if metadata.get(key) != expected:
                            raise DualOutcomeError(
                                f"{metadata_path}: {key}={metadata.get(key)!r}, "
                                f"expected {expected!r}"
                            )

                detailed_path = leaf / "detailed_results.csv"
                detailed_rows = read_detailed_rows(detailed_path, expected_episodes)
                outcomes = {source: [] for source in OUTCOME_COLUMNS}
                for episode, detailed_row in enumerate(detailed_rows):
                    try:
                        expected_length = int(float(detailed_row["length"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise DualOutcomeError(
                            f"{detailed_path}: invalid episode length"
                        ) from exc
                    trace_path = leaf / f"episode_{episode}_controller.csv"
                    traces = read_trace(trace_path, expected_length)
                    computed = {
                        source: glucose_metrics(values)
                        for source, values in traces.items()
                    }
                    compare_canonical_metrics(
                        detailed_row=detailed_row,
                        computed=computed[canonical_source],
                        path=trace_path,
                        tolerance=canonical_tolerance,
                    )
                    for source in OUTCOME_COLUMNS:
                        outcomes[source].append(computed[source])
                    episode_rows += 1
                cell_episode_metrics[(cohort, patient_number, policy_seed)] = outcomes

    expected_cells = len(COHORTS) * len(PATIENT_NUMBERS) * len(POLICY_SEEDS)
    if len(cell_episode_metrics) != expected_cells:
        raise DualOutcomeError(
            f"expected {expected_cells} cells, found {len(cell_episode_metrics)}"
        )

    final: dict[str, dict[str, MetricResult]] = {}
    for source in OUTCOME_COLUMNS:
        final[source] = {}
        for metric in METRICS:
            training_seed_aggregates: list[float] = []
            for policy_seed in POLICY_SEEDS:
                cohort_means: list[float] = []
                for cohort in COHORTS:
                    patient_means: list[float] = []
                    for patient_number in PATIENT_NUMBERS:
                        episodes = cell_episode_metrics[
                            (cohort, patient_number, policy_seed)
                        ][source]
                        patient_means.append(
                            statistics.fmean(episode[metric] for episode in episodes)
                        )
                    cohort_means.append(statistics.fmean(patient_means))
                training_seed_aggregates.append(statistics.fmean(cohort_means))
            final[source][metric] = MetricResult(
                mean=statistics.fmean(training_seed_aggregates),
                sample_sd=statistics.stdev(training_seed_aggregates),
            )
    return DualOutcomeResult(metrics=final, episode_rows=episode_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate true-BG and exposed-CGM metrics from controller CSVs."
    )
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument(
        "--condition", choices=("no_shield", "predictive"), required=True
    )
    parser.add_argument("--logit-penalty", type=float, default=10.0)
    parser.add_argument(
        "--canonical-source",
        choices=tuple(OUTCOME_COLUMNS),
        required=True,
        help="Source used by detailed_results.csv; checked against recomputation.",
    )
    parser.add_argument(
        "--require-baseline-dual-metadata",
        action="store_true",
        help="Require metadata written by the dual-trace paper-baseline job.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = aggregate_dual_outcomes(
            args.result_root,
            expected_episodes=args.expected_episodes,
            condition=args.condition,
            logit_penalty=args.logit_penalty,
            canonical_source=args.canonical_source,
            require_baseline_dual_metadata=args.require_baseline_dual_metadata,
        )
    except DualOutcomeError as exc:
        print(f"Dual-outcome aggregation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "episode_rows": result.episode_rows,
            "outcome_sources": {
                source: {
                    metric: {
                        "mean": values.mean,
                        "sample_sd": values.sample_sd,
                    }
                    for metric, values in metrics.items()
                }
                for source, metrics in result.metrics.items()
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Episode rows: {result.episode_rows}")
        for source, metrics in result.metrics.items():
            print(f"\nOutcome source: {source}")
            for metric in METRICS:
                values = metrics[metric]
                print(
                    f"{metric}: mean={values.mean:.6f}, "
                    f"sample SD={values.sample_sd:.6f}"
                )
        print("\nDual-outcome aggregation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
