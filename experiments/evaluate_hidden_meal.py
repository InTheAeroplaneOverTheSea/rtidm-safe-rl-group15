"""Run paired clean/hidden-meal conditions with the group's policy and Shield."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
if sys.platform == "win32":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "external" / "GlucoAlg"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="adolescent#010")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=22)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--meal-minute", type=int, default=None)
    parser.add_argument("--meal-grams", type=float, default=None)
    parser.add_argument("--hyper-check-threshold", type=float, default=170.0)
    parser.add_argument("--logit-penalty", type=float, choices=[5, 10, 15], default=10.0)
    parser.add_argument(
        "--perturbation-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "perturbations" / "hidden_meal.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "hidden_meal",
    )
    return parser.parse_args()


def _load_meal_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    import yaml

    with args.perturbation_config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("Expected a version-1 hidden-meal configuration.")
    if config.get("observation", []):
        raise ValueError("This paired experiment requires unchanged observations.")
    physiology = config.get("physiology", {})
    if physiology.get("type") != "hidden_meal":
        raise ValueError("Expected physiology.type: hidden_meal.")
    parameters = dict(physiology["parameters"])
    if args.meal_minute is not None:
        parameters["minute"] = args.meal_minute
    if args.meal_grams is not None:
        parameters["grams"] = args.meal_grams
    return parameters


def main() -> None:
    args = _parse_args()
    parameters = _load_meal_parameters(args)

    import eval_run
    from rtidm_safe_rl.perturbations.hidden_meal import (
        HiddenMealCondition,
        HiddenMealEnv,
        verify_hidden_meal_pair,
    )

    meal = HiddenMealCondition(**parameters)
    if meal.grams <= 0 or args.num_episodes < 1:
        raise ValueError("Use positive meal grams and a positive episode count.")
    if not 0.0 < args.hyper_check_threshold < 180.0:
        raise ValueError("Hyper threshold must be between 0 and 180 mg/dL.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh --output-dir; {output_dir} already exists.")
    simulator_source = (
        REPOSITORY_ROOT / "external" / "GlucoSim" / "glucosim"
        / "simglucose" / "sim" / "step.py"
    )
    if "'hidden_meals'" not in simulator_source.read_text(encoding="utf-8"):
        raise RuntimeError("Apply patches/hidden_meal.patch to external/GlucoSim first.")

    output_dir.mkdir(parents=True)
    run_tag = hashlib.sha256(str(output_dir).encode()).hexdigest()[:8]
    original_factory = eval_run.create_diabetes_env
    traces = {}
    summary_rows = []
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPOSITORY_ROOT / "external" / "GlucoAlg")
        for shield_enabled in (False, True):
            for grams in (0.0, meal.grams):
                prefix = "shield" if shield_enabled else "no_shield"
                label = f"{prefix}__{'meal' if grams else 'clean'}"
                condition = HiddenMealCondition(meal.minute, grams, run_tag=run_tag)
                wrappers = []

                def create_env(*factory_args: Any, **factory_kwargs: Any) -> HiddenMealEnv:
                    wrapper = HiddenMealEnv(
                        original_factory(*factory_args, **factory_kwargs),
                        minute=meal.minute,
                        grams=grams,
                    )
                    wrappers.append(wrapper)
                    return wrapper

                eval_run.create_diabetes_env = create_env
                shield_type = args.patient if shield_enabled else "none"
                tag = eval_run._shield_output_tag(
                    shield_type,
                    args.logit_penalty,
                    condition.output_tag,
                    args.hyper_check_threshold,
                )
                result_dir = (
                    Path("diabetes_evaluation") / "t1d" / "CPO" / args.patient
                    / f"seed{args.policy_seed}" / tag
                )
                if result_dir.exists():
                    raise FileExistsError(f"Condition outputs already exist: {result_dir}")
                print(
                    f"Running {label}: {args.num_episodes} seven-day episode(s)...",
                    flush=True,
                )
                eval_run.main(
                    patient_type="t1d",
                    algorithm="CPO",
                    patient_name=args.patient,
                    seed=args.policy_seed,
                    epoch=2441,
                    num_eval_episodes=args.num_episodes,
                    shield_type=shield_type,
                    logit_penalty=args.logit_penalty,
                    observation_perturbation=condition,
                    hyper_check_threshold=args.hyper_check_threshold,
                    eval_seed=args.eval_seed,
                    top_k_bolus_levels=2,
                    context_examples=20,
                )
                traces[label] = wrappers[0].traces
                condition_dir = output_dir / label
                shutil.copytree(result_dir, condition_dir)
                events = [
                    {
                        "episode": episode,
                        "minute": meal.minute,
                        "grams": grams,
                        "ingestion_interval_reached": (
                            trace["meal_delivered"] if grams else
                            any(t >= meal.minute for t in trace["minutes"])
                        ),
                    }
                    for episode, trace in enumerate(traces[label])
                ]
                (condition_dir / "physiology_events.json").write_text(
                    json.dumps(events, indent=2), encoding="utf-8",
                )
                with (result_dir / "summary_results.csv").open(
                    encoding="utf-8", newline="",
                ) as summary_file:
                    for row in csv.DictReader(summary_file):
                        summary_rows.append({"condition": label, **row})
                print(f"Finished {label}. Results: {condition_dir}", flush=True)

        for prefix in ("no_shield", "shield"):
            verify_hidden_meal_pair(
                traces[f"{prefix}__clean"],
                traces[f"{prefix}__meal"],
                minute=meal.minute,
            )
            print(f"{prefix}: pre-meal observations and actions match.", flush=True)
        with (output_dir / "summary.csv").open(
            "w", encoding="utf-8", newline="",
        ) as summary_file:
            writer = csv.DictWriter(
                summary_file, fieldnames=["condition", "Metric", "Value"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        metadata = {
            **{key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()},
            "meal": meal.metadata()["physiology"],
            "horizon_days": 7,
            "premeal_checks_passed": True,
            "top_k_bolus_levels": 2,
            "context_examples": 20,
            "threshold_status": "group smoke setting; not a calibrated reproduction",
        }
        (output_dir / "experiment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        print(f"Comparison saved: {output_dir / 'summary.csv'}", flush=True)
    finally:
        eval_run.create_diabetes_env = original_factory
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
