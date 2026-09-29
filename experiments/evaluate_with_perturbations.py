"""Run GlucoAlg evaluation with a project-owned perturbation configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "external" / "GlucoAlg"))

import eval_run  # noqa: E402

from rtidm_safe_rl.perturbations import load_perturbation_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a diabetes policy with a configurable observation perturbation.",
    )
    parser.add_argument("patient_type", choices=["t1d", "t2d", "t2d_no_pump"])
    parser.add_argument("algorithm")
    parser.add_argument("patient_name")
    parser.add_argument("seed", type=int)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--shield", action="store_true")
    parser.add_argument(
        "--shield-type",
        choices=["predictive", "rule_based", "none"],
        default=None,
    )
    parser.add_argument("--logit-penalty", type=float, default=10.0)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    parser.add_argument(
        "--hyper-check-threshold",
        type=float,
        default=None,
        help="Required for predictive shielding; calibrated upper threshold below 180 mg/dL",
    )
    parser.add_argument("--top-k-bolus-levels", type=int, default=2)
    parser.add_argument("--context-examples", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=22)
    parser.add_argument(
        "--perturbation-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "perturbations" / "none.yaml",
        help="YAML/JSON observation perturbation pipeline",
    )
    return parser.parse_args()


def _resolve_shield_type(args: argparse.Namespace) -> str:
    if args.shield_type == "none":
        return "none"
    if args.shield or args.shield_type is not None:
        if args.shield_type in (None, "predictive"):
            return args.patient_name
        if args.shield_type == "rule_based":
            return "rule_based"
    return "none"


def main() -> None:
    args = _parse_args()
    perturbation = load_perturbation_pipeline(args.perturbation_config)
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPOSITORY_ROOT / "external" / "GlucoAlg")
        eval_run.main(
            patient_type=args.patient_type,
            algorithm=args.algorithm,
            patient_name=args.patient_name,
            seed=args.seed,
            epoch=args.epoch,
            num_eval_episodes=args.num_episodes,
            render=args.render,
            shield_type=_resolve_shield_type(args),
            save_plots=args.save_plots,
            logit_penalty=args.logit_penalty,
            dynamics_model_dir=args.dynamics_model_dir,
            observation_perturbation=perturbation,
            hyper_check_threshold=args.hyper_check_threshold,
            top_k_bolus_levels=args.top_k_bolus_levels,
            context_examples=args.context_examples,
            eval_seed=args.eval_seed,
        )
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
