from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glucosim
from glucosim import gym_env as gym


PATIENT = "adult#001"
SEED = 0

SIMULATION_MINUTES = 24 * 60
SAMPLE_TIME = 5

SHIFT_MINUTES = 60

OUT_DIR = Path("outputs/meal_shift")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_env():
    return gym.make(
        "t1d-v0",
        simulation_minutes=SIMULATION_MINUTES,
        sample_time=SAMPLE_TIME,
        patient_name=PATIENT,
    )


def shift_scenario(env, shift_minutes):
    """
    Shift all valid scenario meals by a fixed amount.

    For this first sanity test we require that no meal
    crosses the 24 h boundary.
    """
    e = env.unwrapped

    scenario = e._jax_state["scenario_meals"]

    times = scenario[..., 0]
    amounts = scenario[..., 1]

    valid = (times >= 0) & (amounts > 0)

    shifted_times = jnp.where(
        valid,
        times + shift_minutes,
        times,
    )

    # First experiment: don't allow day rollover yet
    valid_shifted_times = shifted_times[valid]

    if valid_shifted_times.size > 0:
        max_time = float(jnp.max(valid_shifted_times))

        if max_time >= 1440:
            raise ValueError(
                f"Meal shift crosses day boundary: "
                f"latest meal = {max_time:.1f} min"
            )

    shifted = scenario.at[..., 0].set(shifted_times)

    # Replace the scenario inside the actual simulator state
    new_state = dict(e._jax_state)
    new_state["scenario_meals"] = shifted
    e._jax_state = new_state

    # Keep wrapper-side copy synchronized for printing/debugging
    e.scenario_meals = shifted


def rollout(env, initial_obs):
    obs = initial_obs

    records = []

    step_idx = 0
    done = False

    while not done:

        records.append(
            {
                "step": step_idx,
                "time_min": step_idx * SAMPLE_TIME,
                "time_hour": step_idx * SAMPLE_TIME / 60.0,
                "bg": float(obs[0]),
                "iob": float(obs[1]),
                "cob": float(obs[2]),
            }
        )

        action = np.array([0, 0], dtype=np.int64)

        obs, reward, cost, terminated, truncated, info = env.step(action)

        step_idx += 1
        done = terminated or truncated

    records.append(
        {
            "step": step_idx,
            "time_min": step_idx * SAMPLE_TIME,
            "time_hour": step_idx * SAMPLE_TIME / 60.0,
            "bg": float(obs[0]),
            "iob": float(obs[1]),
            "cob": float(obs[2]),
        }
    )

    return pd.DataFrame(records)

# =========================
# Baseline
# =========================

baseline_env = make_env()

baseline_obs, _ = baseline_env.reset(seed=SEED)

print("\n===== Nominal scenario =====")
baseline_env.unwrapped._print_scenario_meals(
    baseline_env.unwrapped.scenario_meals
)

baseline_df = rollout(
    baseline_env,
    baseline_obs,
)

baseline_env.close()

# =========================
# +60 min meal shift
# =========================

stress_env = make_env()

stress_obs, _ = stress_env.reset(seed=SEED)

# Before shifting, this should be IDENTICAL
# to the baseline schedule.
print("\n===== Scenario before stress =====")
stress_env.unwrapped._print_scenario_meals(
    stress_env.unwrapped.scenario_meals
)

shift_scenario(
    stress_env,
    SHIFT_MINUTES,
)

print(f"\n===== Scenario after +{SHIFT_MINUTES} min shift =====")
stress_env.unwrapped._print_scenario_meals(
    stress_env.unwrapped.scenario_meals
)

stress_df = rollout(
    stress_env,
    stress_obs,
)

stress_env.close()

baseline_df.to_csv(
    OUT_DIR / "baseline.csv",
    index=False,
)

stress_df.to_csv(
    OUT_DIR / f"meal_shift_{SHIFT_MINUTES}min.csv",
    index=False,
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
)


# =========================
# BG
# =========================

axes[0].plot(
    baseline_df["time_hour"],
    baseline_df["bg"],
    label="Nominal",
    linewidth=2,
)

axes[0].plot(
    stress_df["time_hour"],
    stress_df["bg"],
    label=f"Meal shift +{SHIFT_MINUTES} min",
    linewidth=2,
)

axes[0].axhline(
    70,
    linestyle="--",
    linewidth=1,
)

axes[0].axhline(
    180,
    linestyle="--",
    linewidth=1,
)

axes[0].set_ylabel("BG (mg/dL)")
axes[0].set_title(
    f"{PATIENT}: nominal vs meal timing shift"
)

axes[0].grid(alpha=0.3)
axes[0].legend()


# =========================
# COB
# =========================

axes[1].plot(
    baseline_df["time_hour"],
    baseline_df["cob"],
    label="Nominal",
    linewidth=2,
)

axes[1].plot(
    stress_df["time_hour"],
    stress_df["cob"],
    label=f"Meal shift +{SHIFT_MINUTES} min",
    linewidth=2,
)

axes[1].set_xlabel("Time (hours)")
axes[1].set_ylabel("COB")

axes[1].set_xlim(0, 24)

axes[1].grid(alpha=0.3)
axes[1].legend()


plt.tight_layout()

plot_path = OUT_DIR / "nominal_vs_shift_60min.png"

plt.savefig(
    plot_path,
    dpi=150,
)

plt.show()

print("\nSaved:", plot_path)

