#!/usr/bin/env python3
"""Validate and aggregate one author-v1 T1D evaluation condition.

The aggregation order is:

    episodes -> patient -> cohort -> policy-training seed -> final result

The input directory is one algorithm root, for example::

    author_v1_all_baselines_evaluation/no_shield/t1d/CUP
    author_v1_all_baselines_evaluation/predictive_penalty10/t1d/CUP

Only the Python standard library is required, so this is safe to run on a
Snellius login node after the Slurm evaluation jobs have finished.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


COHORTS = ("child", "adolescent", "adult")
POLICY_SEEDS = (0, 1, 2)
PATIENT_NUMBERS = tuple(range(2, 11))
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
METRICS = {
    "TIR (%)": "time_in_range_pct",
    "Risk index": "risk_index",
    "CV (%)": "cv_pct",
}
EXPECTED_METADATA = {
    "code_stack": "author_v1",
    "glucoalg_commit": "50d3134fd6f0e01019295be0d6b99f110f794753",
    "glucosim_commit": "6522d49ee1e6c1fd34f287519a5d41df735bbd38",
    "evaluation_seed": "22",
    "checkpoint_epoch": "2441",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the 3x3x9 author-v1 T1D matrix.",
    )
    parser.add_argument(
        "result_root",
        type=Path,
        help=(
            "One algorithm directory containing patient/seed/condition "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        required=True,
        help="Expected episode count in every patient/checkpoint cell.",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        help=(
            "Algorithm to require in run_metadata.txt. If omitted, the name of "
            "result_root is used for display but missing algorithm metadata is "
            "accepted for compatibility with the earlier CPO-only job."
        ),
    )
    parser.add_argument(
        "--condition",
        choices=("no_shield", "predictive"),
        default="no_shield",
        help="Condition represented by result_root (default: no_shield).",
    )
    parser.add_argument(
        "--logit-penalty",
        type=float,
        default=10.0,
        help="Predictive-Shield logit penalty and output suffix (default: 10).",
    )
    parser.add_argument(
        "--expected-result-collection",
        help=(
            "If set, require this result_collection value in every metadata "
            "file. Team-wide runs normally use "
            "author_v1_all_baselines_evaluation."
        ),
    )
    parser.add_argument(
        "--paper-tir-mean",
        type=float,
        help="Optional paper TIR mean to print a numerical comparison.",
    )
    parser.add_argument(
        "--paper-tir-std",
        type=float,
        help="Optional paper TIR standard deviation to compare.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    metadata: dict[str, str] = {}
    hash_records: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
        elif line.strip():
            fields = line.split(maxsplit=1)
            digest = fields[0]
            source = fields[1] if len(fields) == 2 else ""
            if len(digest) == 64 and all(
                char in "0123456789abcdef" for char in digest
            ):
                hash_records.append((digest, source))
    return metadata, hash_records


def number_tag(value: float) -> str:
    """Format a numeric CLI value the same way the Slurm job names outputs."""

    return str(int(value)) if value.is_integer() else str(value)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    args = parse_args()
    if args.expected_episodes < 1:
        raise SystemExit("--expected-episodes must be at least 1")
    if (args.paper_tir_mean is None) != (args.paper_tir_std is None):
        raise SystemExit(
            "--paper-tir-mean and --paper-tir-std must be provided together"
        )

    root = args.result_root.resolve()
    algorithm = args.algorithm or root.name
    penalty_tag = number_tag(args.logit_penalty)
    result_tag = (
        "no_shield"
        if args.condition == "no_shield"
        else f"with_shield_{penalty_tag}"
    )
    errors: list[str] = []
    cells: dict[tuple[str, int, int], list[dict[str, float]]] = {}
    checkpoint_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    predictor_hashes: set[str] = set()

    for cohort in COHORTS:
        for patient_number in PATIENT_NUMBERS:
            patient_name = f"{cohort}#{patient_number:03d}"
            for policy_seed in POLICY_SEEDS:
                result_dir = root / patient_name / f"seed{policy_seed}" / result_tag
                summary_path = result_dir / "summary_results.csv"
                detail_path = result_dir / "detailed_results.csv"
                metadata_path = result_dir / "run_metadata.txt"
                cell = (cohort, patient_number, policy_seed)

                if not summary_path.is_file():
                    fail(errors, f"missing {summary_path}")
                if not detail_path.is_file():
                    fail(errors, f"missing {detail_path}")
                    continue
                if not metadata_path.is_file():
                    fail(errors, f"missing {metadata_path}")
                    continue

                with detail_path.open(newline="") as handle:
                    raw_rows = list(csv.DictReader(handle))
                if len(raw_rows) != args.expected_episodes:
                    fail(
                        errors,
                        f"{detail_path}: expected {args.expected_episodes} episodes, "
                        f"found {len(raw_rows)}",
                    )

                episode_ids = [int(row["episode"]) for row in raw_rows]
                if episode_ids != list(range(len(raw_rows))):
                    fail(errors, f"{detail_path}: unexpected episode IDs {episode_ids}")

                numeric_rows: list[dict[str, float]] = []
                for row in raw_rows:
                    try:
                        numeric_rows.append(
                            {column: float(row[column]) for column in METRICS.values()}
                            | {"length": float(row["length"])}
                        )
                    except (KeyError, ValueError) as exc:
                        fail(errors, f"{detail_path}: invalid metric row ({exc})")
                cells[cell] = numeric_rows

                metadata, hash_records = read_metadata(metadata_path)
                expected = EXPECTED_METADATA | {
                    "condition": args.condition,
                    "episodes": str(args.expected_episodes),
                }
                if args.algorithm is not None:
                    expected["algorithm"] = args.algorithm
                if args.expected_result_collection is not None:
                    expected["result_collection"] = args.expected_result_collection
                if args.condition == "predictive":
                    expected |= {
                        "predictor_provenance": (
                            "group_trained_ba_node_b5_p24_h24_seed42"
                        ),
                        "v1_use_meal_hyper_check": "false",
                        "v1_hyper_check_threshold_mgdl": "250",
                        "logit_penalty": penalty_tag,
                    }
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        fail(
                            errors,
                            f"{metadata_path}: {key}={metadata.get(key)!r}, expected {value!r}",
                        )
                if metadata.get("patient_name") != patient_name:
                    fail(errors, f"{metadata_path}: patient_name mismatch")
                if metadata.get("policy_seed") != str(policy_seed):
                    fail(errors, f"{metadata_path}: policy_seed mismatch")

                policy_hashes = [
                    digest
                    for digest, source in hash_records
                    if source.endswith("/torch_save/epoch-2441.pt")
                ]
                if len(policy_hashes) != 1:
                    fail(errors, f"{metadata_path}: missing checkpoint SHA-256")
                else:
                    checkpoint_hashes[(cohort, policy_seed)].add(policy_hashes[0])

                if args.condition == "predictive":
                    cell_predictor_hashes = [
                        digest
                        for digest, source in hash_records
                        if source.endswith("/best_model.pth")
                    ]
                    if len(cell_predictor_hashes) != 1:
                        fail(errors, f"{metadata_path}: missing predictor SHA-256")
                    else:
                        predictor_hashes.add(cell_predictor_hashes[0])

    expected_cells = len(COHORTS) * len(PATIENT_NUMBERS) * len(POLICY_SEEDS)
    total_episodes = sum(len(rows) for rows in cells.values())
    print(f"Algorithm: {algorithm}")
    print(f"Condition: {args.condition}")
    print(f"Result root: {root}")
    print(f"Matrix: {len(cells)}/{expected_cells} cells, {total_episodes} episode rows")

    for key, hashes in sorted(checkpoint_hashes.items()):
        if len(hashes) != 1:
            fail(errors, f"{key}: inconsistent checkpoint hashes {sorted(hashes)}")
    print(f"Checkpoint identities: {len(checkpoint_hashes)}/9 cohort/seed pairs")
    if args.condition == "predictive":
        if len(predictor_hashes) != 1:
            fail(
                errors,
                f"expected one predictor identity, found {sorted(predictor_hashes)}",
            )
        print(f"Predictor identities: {len(predictor_hashes)}/1")

    if errors:
        print("\nValidation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Validation: PASS")
    early_terminated = sum(
        row["length"] < 2016 for rows in cells.values() for row in rows
    )
    print(f"Episodes shorter than 2016 steps: {early_terminated}/{total_episodes}")

    final_by_metric: dict[str, tuple[float, float, float]] = {}
    for label, column in METRICS.items():
        print(f"\n{label}")
        print("policy_seed,child,adolescent,adult,training_seed_aggregate")
        training_seed_aggregates: list[float] = []
        for policy_seed in POLICY_SEEDS:
            cohort_means: list[float] = []
            display_values: list[str] = []
            for cohort in COHORTS:
                patient_means = []
                for patient_number in PATIENT_NUMBERS:
                    rows = cells[(cohort, patient_number, policy_seed)]
                    patient_means.append(statistics.mean(row[column] for row in rows))
                cohort_mean = statistics.mean(patient_means)
                cohort_means.append(cohort_mean)
                display_values.append(f"{cohort_mean:.6f}")
            training_seed_mean = statistics.mean(cohort_means)
            training_seed_aggregates.append(training_seed_mean)
            print(
                f"{policy_seed},{','.join(display_values)},{training_seed_mean:.6f}"
            )

        final_mean = statistics.mean(training_seed_aggregates)
        sample_sd = statistics.stdev(training_seed_aggregates)
        population_sd = statistics.pstdev(training_seed_aggregates)
        final_by_metric[label] = (final_mean, sample_sd, population_sd)
        print(
            f"final mean={final_mean:.6f}, sample SD={sample_sd:.6f}, "
            f"population SD={population_sd:.6f}"
        )

    if args.paper_tir_mean is not None and args.paper_tir_std is not None:
        tir_mean, tir_sample_sd, tir_population_sd = final_by_metric["TIR (%)"]
        print("\nPaper comparison (TIR %)")
        print(
            f"reported={args.paper_tir_mean:.2f} +/- {args.paper_tir_std:.2f}; "
            f"reproduced mean delta={tir_mean - args.paper_tir_mean:+.6f}; "
            f"sample-SD delta={tir_sample_sd - args.paper_tir_std:+.6f}; "
            f"population-SD delta={tir_population_sd - args.paper_tir_std:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
