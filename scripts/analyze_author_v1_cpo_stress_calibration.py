#!/usr/bin/env python3
"""Validate and summarize the one-cell CPO stress calibration on SURF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


SCENARIOS = (
    "nominal",
    "gaussian_sigma15",
    "bias_positive",
    "bias_negative",
    "drift_sigma0025_calibration",
    "random_walk_drift",
    "drift_sigma004_calibration",
    "dropout_markov_v1",
    "unannounced_meal_60g",
)

DEFAULT_SENSOR_SCENARIOS = tuple(
    scenario for scenario in SCENARIOS if scenario != "unannounced_meal_60g"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("author_v1_stress_calibration"),
    )
    parser.add_argument("--patient", default="child#002")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--replicate-id", type=int, default=0)
    parser.add_argument("--base-evaluation-seed", type=int, default=22)
    parser.add_argument("--logit-penalty", type=float, default=10.0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=DEFAULT_SENSOR_SCENARIOS,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def number_tag(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def leaf_path(args: argparse.Namespace, scenario: str, condition: str) -> Path:
    if condition == "no_shield":
        result_condition = "no_shield"
        source_tag = "no_shield"
    else:
        result_condition = f"predictive_penalty{number_tag(args.logit_penalty)}"
        source_tag = f"with_shield_{number_tag(args.logit_penalty)}"
    return (
        args.root
        / scenario
        / f"replicate{args.replicate_id}"
        / result_condition
        / "t1d"
        / "CPO"
        / args.patient
        / f"seed{args.policy_seed}"
        / source_tag
    )


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {value}")
    return value


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def run_lengths(values: list[bool]) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in values + [False]:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    return lengths


def summarize_leaf(
    args: argparse.Namespace, scenario: str, condition: str, errors: list[str]
) -> dict:
    leaf = leaf_path(args, scenario, condition)
    metadata_path = leaf / "stress_episode_metadata.json"
    detailed_path = leaf / "detailed_results.csv"
    if not metadata_path.is_file():
        errors.append(f"missing {metadata_path}")
        return {}
    if not detailed_path.is_file():
        errors.append(f"missing {detailed_path}")
        return {}

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {metadata_path}: {exc}")
        return {}

    if metadata.get("schema_version") != 2:
        errors.append(f"{metadata_path}: schema_version is not 2")
    if metadata.get("stress_scenario") != scenario:
        errors.append(f"{metadata_path}: scenario mismatch")
    if metadata.get("outcome_source") != "true physiological BG_hist average":
        errors.append(f"{metadata_path}: outcome source is not true BG")
    episodes = metadata.get("episodes", [])
    expected_seeds = [
        args.base_evaluation_seed + args.replicate_id * 10_000 + episode
        for episode in range(args.episodes)
    ]
    actual_seeds = [record.get("episode_seed") for record in episodes]
    if actual_seeds != expected_seeds:
        errors.append(
            f"{metadata_path}: expected seeds {expected_seeds}, found {actual_seeds}"
        )
    if scenario == "unannounced_meal_60g":
        expected_hidden = [
            {
                "day": day,
                "time_minutes": 900,
                "absolute_time_minutes": (day - 1) * 1440 + 900,
                "carbs_g": 60.0,
            }
            for day in range(1, 8)
        ]
        for record in episodes:
            if record.get("planned_unannounced_meals") != expected_hidden:
                errors.append(
                    f"{metadata_path}: unexpected hidden-meal plan in "
                    f"episode {record.get('episode')}"
                )

    true_bg: list[float] = []
    exposed_cgm: list[float] = []
    native_cgm: list[float] = []
    scales: list[float] = []
    injected: list[bool] = []
    gap_lengths: list[int] = []
    native_dropout_minutes = 0
    sensor_clip_minutes = 0
    scale_boundary_minutes = 0
    exposed_clip_observations = 0

    for episode in range(args.episodes):
        trace_path = leaf / f"episode_{episode}_controller.csv"
        if not trace_path.is_file():
            errors.append(f"missing {trace_path}")
            continue
        episode_injected: list[bool] = []
        with trace_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "BG",
                "CGM",
                "Native_CGM",
                "CGM_Trend",
                "CGM_Scale",
                "CGM_Scale_Boundary_Count",
                "Native_Dropout_Count",
                "Injected_Markov_Dropout",
                "Sensor_Clip_Count",
                "Exposed_Sensor_Clipped",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                errors.append(f"{trace_path}: missing columns {sorted(missing)}")
                continue
            for row in reader:
                try:
                    true_bg.append(as_float(row, "BG"))
                    exposed_cgm.append(as_float(row, "CGM"))
                    native_cgm.append(as_float(row, "Native_CGM"))
                    scales.append(as_float(row, "CGM_Scale"))
                    scale_boundary_minutes += int(
                        float(row["CGM_Scale_Boundary_Count"])
                    )
                    native_dropout_minutes += int(float(row["Native_Dropout_Count"]))
                    sensor_clip_minutes += int(float(row["Sensor_Clip_Count"]))
                    gap_value = as_bool(row["Injected_Markov_Dropout"])
                    injected.append(gap_value)
                    episode_injected.append(gap_value)
                    exposed_clip_observations += int(
                        as_bool(row["Exposed_Sensor_Clipped"])
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{trace_path}: invalid diagnostic row ({exc})")
                    break
        gap_lengths.extend(run_lengths(episode_injected))

    if not true_bg:
        return {}
    errors_mgdl = [cgm - bg for cgm, bg in zip(exposed_cgm, true_bg)]
    delivered_meals = sum(
        len(record.get("delivered_unannounced_meals", [])) for record in episodes
    )
    return {
        "scenario": scenario,
        "condition": condition,
        "observations": len(true_bg),
        "episode_seeds": actual_seeds,
        "sensor_error_mean_mgdl": statistics.fmean(errors_mgdl),
        "sensor_error_sd_mgdl": statistics.pstdev(errors_mgdl),
        "sensor_error_rmse_mgdl": math.sqrt(
            statistics.fmean(value * value for value in errors_mgdl)
        ),
        "cgm_scale_min": min(scales),
        "cgm_scale_max": max(scales),
        "cgm_scale_boundary_minutes": scale_boundary_minutes,
        "cgm_scale_boundary_pct": (
            100.0 * scale_boundary_minutes / (5.0 * len(scales))
        ),
        "native_dropout_minutes": native_dropout_minutes,
        "sensor_clip_minutes": sensor_clip_minutes,
        "exposed_clip_pct": 100.0 * exposed_clip_observations / len(true_bg),
        "markov_gap_count": len(gap_lengths),
        "markov_gap_mean_minutes": (
            statistics.fmean(gap_lengths) * 5.0 if gap_lengths else 0.0
        ),
        "markov_missing_pct": 100.0 * sum(injected) / len(injected),
        "delivered_unannounced_meals": delivered_meals,
    }


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    summaries = [
        summarize_leaf(args, scenario, condition, errors)
        for scenario in args.scenarios
        for condition in ("no_shield", "predictive")
    ]
    summaries = [summary for summary in summaries if summary]

    print(
        "scenario\tcondition\tn\terror_mean\terror_sd\terror_rmse\t"
        "scale_min\tscale_max\tscale_boundary_pct\tclip_pct\t"
        "gap_count\tgap_mean_min\tmissing_pct\tmeals"
    )
    for item in summaries:
        print(
            f"{item['scenario']}\t{item['condition']}\t{item['observations']}\t"
            f"{item['sensor_error_mean_mgdl']:.3f}\t"
            f"{item['sensor_error_sd_mgdl']:.3f}\t"
            f"{item['sensor_error_rmse_mgdl']:.3f}\t"
            f"{item['cgm_scale_min']:.4f}\t{item['cgm_scale_max']:.4f}\t"
            f"{item['cgm_scale_boundary_pct']:.3f}\t"
            f"{item['exposed_clip_pct']:.3f}\t"
            f"{item['markov_gap_count']}\t"
            f"{item['markov_gap_mean_minutes']:.2f}\t"
            f"{item['markov_missing_pct']:.3f}\t"
            f"{item['delivered_unannounced_meals']}"
        )

    output = args.output or (args.root / "cpo_calibration_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "PASS" if not errors else "FAIL",
                "selection_rule": (
                    "Validate the fixed hidden-meal plan before inspecting "
                    "Shield outcome differences."
                    if args.scenarios == ["unannounced_meal_60g"]
                    else "Freeze severity from sensor diagnostics before inspecting "
                    "Shield outcome differences."
                ),
                "summaries": summaries,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {output}")
    if errors:
        print("Calibration validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Calibration structure and diagnostics: PASS")
    if any("drift" in scenario for scenario in args.scenarios):
        print("Drift severity still requires human review of the printed diagnostics.")
    else:
        print("Requested scenario set validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
