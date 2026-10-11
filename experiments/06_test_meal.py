"""Run GlucoSim meal-stress rollouts and plot actual BG / CGM trajectories.

Prerequisites:
    - GlucoSim is installed into the active environment.
    - The previously supplied ``interface_modified.py`` is in this directory
      or in its parent directory.
    - matplotlib and numpy are installed.

PowerShell example (from C:\\Tue\\2AMM20):
    uv run python .\\experiments\\06_test_meal.py --show
    uv run python .\\experiments\\06_test_meal.py --days 7

IMPORTANT:
    This is an environment / meal-schedule smoke test using a FIXED action
    [bolus_level=0, meal_level=0]. It is *not* a trained OnCRPO policy and
    does not evaluate predictive shielding. Basal insulin continues to operate.
    Unsafe glucose values or early termination are possible; this is a
    simulation-only experiment, not clinical advice.

The plot shows BG and CGM measured from the simulator state AFTER each step.
The vertical meal lines indicate PLANNED meal times, not confirmed intake.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent):
    if (path / "interface.py").is_file():
        sys.path.insert(0, str(path))
        break
else:
    raise FileNotFoundError(
        "Cannot find interface_modified.py next to 06_test_meal.py "
        "or in the parent project folder."
    )

from interface import GlucoSimInterface, Meal, StressConfig

# ---------------------------------------------------------------------------
# Default configuration: change these values or use the command-line options.
# ---------------------------------------------------------------------------
DEFAULT_PATIENT = "adult#001"
DEFAULT_DAYS = 1        # quick visual sanity check; try --days 7 later
SAMPLE_TIME = 5         # minutes per action
DEFAULT_SEED = 42

BASE_MEALS = [
    Meal("07:00", 45),
    Meal("12:30", 70),
    Meal("18:00", 80),
    Meal("21:00", 15),
]  # total 210 g/day

# Five meals, keeping the daily total at 210 g to isolate meal frequency.
FIVE_MEALS = [
    Meal("07:00", 35),
    Meal("10:30", 10),
    Meal("12:30", 60),
    Meal("18:00", 90),
    Meal("21:00", 15),
]


def make_stresses() -> list[tuple[str, StressConfig]]:
    """Return paired scenarios with identical nominal seeds and schedules."""
    return [
        ("Nominal", StressConfig(daily_meals=BASE_MEALS)),
        ("Meal timing +60 min", StressConfig(
            daily_meals=BASE_MEALS,
            meal_shift_minutes=60,
        )),
        ("Meal amount x1.25", StressConfig(
            daily_meals=BASE_MEALS,
            meal_amount_scale=1.25,
        )),
        ("5 meals/day (same total CHO)", StressConfig(
            daily_meals=FIVE_MEALS,
        )),
        ("Combined: +60 min, x1.25, 5 meals", StressConfig(
            daily_meals=FIVE_MEALS,
            meal_shift_minutes=60,
            meal_amount_scale=1.25,
        )),
    ]


def extract_bg_and_cgm(env) -> tuple[float, float]:
    """Read 5-minute BG and CGM history at the current index.

    Verified against GlucoSim revision ddd674e...:
        reset(): BG_hist[0] and CGM_hist[0] hold initial measurements.
        step():  BG_hist[index] and CGM_hist[index] receive the just-finished
                 sample-interval means; the returned state's index increments.

    Do not use observation[0] as true BG: it is the CGM measurement.
    """
    s = env.unwrapped._jax_state
    idx = int(np.asarray(s["index"])) - 1
    bg = float(np.asarray(s["BG_hist"][idx]))
    cgm = float(np.asarray(s["CGM_hist"][idx]))
    return bg, cgm


def rollout(
    interface: GlucoSimInterface,
    stress: StressConfig,
    *,
    seed: int,
    action: np.ndarray,
) -> dict:
    """Run one episode without a learned controller, and capture actual states."""
    env, obs, info = interface.create(stress=stress, seed=seed)
    try:
        planned_meals = interface.get_meal_events(env)
        initial_bg, initial_cgm = extract_bg_and_cgm(env)
        bg_history = [initial_bg]
        cgm_history = [initial_cgm]
        times_min = [0.0]
        rewards = []
        costs = []
        scheduled_meal_signals = []
        termination_reason = "step_limit"
        max_steps = interface.simulation_minutes // interface.sample_time

        for step_idx in range(max_steps):
            # GlucoSim's Gym adapter returns 6 values: reward and cost separate.
            obs, reward, cost, terminated, truncated, info = env.step(action)
            bg, cgm = extract_bg_and_cgm(env)
            bg_history.append(bg)
            cgm_history.append(cgm)
            times_min.append(float(np.asarray(env.unwrapped._jax_state["t"])))
            rewards.append(float(reward))
            costs.append(float(cost))
            scheduled_meal_signals.append(float(info.get("scenario_meal_avg", 0.0)))

            if terminated or truncated:
                termination_reason = str(info.get(
                    "_last_termination_cause",
                    "terminated" if terminated else "time_limit",
                ))
                break

        return {
            "time_h": np.asarray(times_min, dtype=float) / 60.0,
            "true_bg": np.asarray(bg_history, dtype=float),
            "cgm": np.asarray(cgm_history, dtype=float),
            "rewards": np.asarray(rewards, dtype=float),
            "costs": np.asarray(costs, dtype=float),
            "scheduled_meal_signals": np.asarray(scheduled_meal_signals, dtype=float),
            "meal_events": planned_meals,
            "steps": len(rewards),
            "termination_reason": termination_reason,
            "observation_shape": tuple(np.shape(obs)),
        }
    finally:
        env.close()


def plot_rollouts(
    cases: list[tuple[str, dict]], days: int, figure_path: Path, show: bool
) -> None:
    """Five stacked panels with the *simulated* BG/CGM time series."""
    fig, axes = plt.subplots(
        len(cases), 1, figsize=(13, 3.0 * len(cases)), sharex=True,
        constrained_layout=True,
    )
    max_h = days * 24
    for ax, (label, result) in zip(np.atleast_1d(axes), cases):
        t = result["time_h"]  # use each episode's actual length
        ax.plot(t, result["true_bg"], label="True BG", linewidth=1.8)
        ax.plot(t, result["cgm"], label="CGM", linewidth=1.0, alpha=0.8)
        ax.axhline(70, linestyle="--", linewidth=0.85, alpha=0.55)
        ax.axhline(180, linestyle="--", linewidth=0.85, alpha=0.55)

        for event in result["meal_events"]:
            meal_hour = ((int(event["day"]) - 1) * 1440 + float(event["time_minutes"])) / 60.0
            ax.axvline(meal_hour, linestyle=":", linewidth=0.75, alpha=0.28)

        ax.set_ylabel("mg/dL")
        ax.set_title(f"{label} | {result['steps']} steps | {result['termination_reason']}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", ncol=2, fontsize=8)

    axes[-1].set_xlabel("Simulation time (hours)")
    axes[-1].set_xlim(0, max_h)
    axes[-1].set_xticks(np.arange(0, max_h + 0.01, 3 if days == 1 else 24))
    fig.suptitle(
        "GlucoSim: true BG vs CGM under meal stresses (fixed action, no shield)",
        fontsize=13,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved: {figure_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_csv(cases: list[tuple[str, dict]], output_file: Path) -> None:
    """Save time series with each case's actual time axis (can terminate early)."""
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "time_hours", "true_bg_mgdl", "cgm_mgdl"])
        for label, result in cases:
            for t, bg, cgm in zip(result["time_h"], result["true_bg"], result["cgm"]):
                writer.writerow([label, float(t), float(bg), float(cgm)])
    print(f"CSV saved: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated BG/CGM for meal stresses")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Simulated days (default: 1)")
    parser.add_argument("--patient", default=DEFAULT_PATIENT, help="GlucoSim patient name")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Reset seed")
    parser.add_argument("--show", action="store_true", help="Display plot in addition to saving")
    parser.add_argument("--bolus-level", type=int, default=0, choices=range(5), metavar="{0..4}")
    parser.add_argument("--meal-level", type=int, default=0, choices=range(5), metavar="{0..4}")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be >= 1")

    interface = GlucoSimInterface(
        patient_name=args.patient,
        simulation_minutes=args.days * 1440,
        sample_time=SAMPLE_TIME,
    )
    action = np.asarray([args.bolus_level, args.meal_level], dtype=np.int32)
    output_dir = HERE / "figures"
    cases = []

    print(f"Patient={args.patient}; days={args.days}; seed={args.seed}; fixed action={action.tolist()}")
    print("NOTE: no trained policy and no predictive shield is used.\n")
    for name, stress in make_stresses():
        print(f"Running: {name} ...", flush=True)
        result = rollout(interface, stress, seed=args.seed, action=action)
        cases.append((name, result))
        meal_count = len(result["meal_events"])
        total_cho = sum(e["carbs_g"] for e in result["meal_events"])
        print(
            f"  {result['steps']} steps; planned meals={meal_count}; "
            f"planned CHO={total_cho:.1f} g; "
            f"BG end={result['true_bg'][-1]:.1f}; "
            f"cumulative cost={np.sum(result['costs']):.1f}; "
            f"end reason={result['termination_reason']}"
        )

    fig_path = output_dir / "meal_stress_bg_cgm.png"
    csv_path = output_dir / "meal_stress_bg_cgm.csv"
    plot_rollouts(cases, args.days, fig_path, args.show)
    save_csv(cases, csv_path)


if __name__ == "__main__":
    main()
