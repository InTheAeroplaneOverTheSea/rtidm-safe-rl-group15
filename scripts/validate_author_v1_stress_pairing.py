#!/usr/bin/env python3
"""Verify that no-Shield and Shield stress cells use identical disturbances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


COHORTS = ("child", "adolescent", "adult")
PATIENT_NUMBERS = tuple(range(2, 11))
POLICY_SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("no_shield_root", type=Path)
    parser.add_argument("predictive_root", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--logit-penalty", type=float, default=10.0)
    return parser.parse_args()


def number_tag(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def comparable_episode(record: dict) -> dict:
    """Fields that must match before the two controllers start to diverge."""
    return {
        "episode": record.get("episode"),
        "episode_seed": record.get("episode_seed"),
        "meal_schedule_sha256": record.get("meal_schedule_sha256"),
        "planned_unannounced_meals": record.get("planned_unannounced_meals"),
        "cgm_obs_sigma_mgdl": record.get("cgm_obs_sigma_mgdl"),
    }


def main() -> int:
    args = parse_args()
    if args.expected_episodes < 1:
        raise SystemExit("--expected-episodes must be positive")
    no_root = args.no_shield_root.resolve()
    shield_root = args.predictive_root.resolve()
    shield_tag = f"with_shield_{number_tag(args.logit_penalty)}"
    errors: list[str] = []
    checked = 0

    for cohort in COHORTS:
        for patient_number in PATIENT_NUMBERS:
            patient = f"{cohort}#{patient_number:03d}"
            for policy_seed in POLICY_SEEDS:
                no_path = (
                    no_root
                    / patient
                    / f"seed{policy_seed}"
                    / "no_shield"
                    / "stress_episode_metadata.json"
                )
                shield_path = (
                    shield_root
                    / patient
                    / f"seed{policy_seed}"
                    / shield_tag
                    / "stress_episode_metadata.json"
                )
                if not no_path.is_file():
                    errors.append(f"missing {no_path}")
                    continue
                if not shield_path.is_file():
                    errors.append(f"missing {shield_path}")
                    continue

                try:
                    no_payload = json.loads(no_path.read_text())
                    shield_payload = json.loads(shield_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"{patient}/seed{policy_seed}: {exc}")
                    continue

                for key in (
                    "stress_scenario",
                    "stress_config",
                    "base_evaluation_seed",
                    "replicate_id",
                    "episode_seed_policy",
                    "unannounced_meal_semantics",
                ):
                    if no_payload.get(key) != shield_payload.get(key):
                        errors.append(
                            f"{patient}/seed{policy_seed}: paired {key} mismatch"
                        )

                no_episodes = no_payload.get("episodes", [])
                shield_episodes = shield_payload.get("episodes", [])
                if len(no_episodes) != args.expected_episodes:
                    errors.append(
                        f"{no_path}: expected {args.expected_episodes} episodes"
                    )
                if len(shield_episodes) != args.expected_episodes:
                    errors.append(
                        f"{shield_path}: expected {args.expected_episodes} episodes"
                    )
                if [comparable_episode(item) for item in no_episodes] != [
                    comparable_episode(item) for item in shield_episodes
                ]:
                    errors.append(
                        f"{patient}/seed{policy_seed}: episode seeds or planned "
                        "disturbances are not paired"
                    )
                checked += 1

    print(f"Paired stress cells: {checked}/81")
    if errors:
        print("Pairing validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Pairing validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
