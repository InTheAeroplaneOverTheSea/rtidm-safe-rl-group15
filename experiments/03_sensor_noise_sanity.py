from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glucosim
from glucosim import gym_env as gym


PATIENT = "adult#001"
SEED = 0

SIMULATION_MINUTES = 24 * 60
SAMPLE_TIME = 5

NOMINAL_SIGMA = 1.5
STRESS_SIGMA = 15.0

OUT_DIR = Path("outputs/sensor_noise")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(cgm_sigma):
    env = gym.make(
        "t1d-v0",
        simulation_minutes=SIMULATION_MINUTES,
        sample_time=SAMPLE_TIME,
        patient_name=PATIENT,
    )

    e = env.unwrapped

    new_noise_config = replace(
        e.env_params.noise_config,
        cgm_obs_sigma_mgdl=cgm_sigma,
    )

    e.env_params = replace(
        e.env_params,
        noise_config=new_noise_config,
    )

    return env


def get_current_values(env, obs):
    e = env.unwrapped

    idx = int(e._jax_state["index"]) - 1

    true_bg = float(e._jax_state["BG_hist"][idx])
    cgm_hist = float(e._jax_state["CGM_hist"][idx])
    observed_cgm = float(obs[0])

    return true_bg, cgm_hist, observed_cgm


def rollout(env, seed):
    obs, _ = env.reset(seed=seed)

    records = []

    step_idx = 0
    done = False

    while not done:

        true_bg, cgm_hist, observed_cgm = get_current_values(
            env,
            obs,
        )

        records.append(
            {
                "step": step_idx,
                "time_min": step_idx * SAMPLE_TIME,
                "time_hour": step_idx * SAMPLE_TIME / 60,
                "true_bg": true_bg,
                "cgm": cgm_hist,
                "obs_cgm": observed_cgm,
                "sensor_error": observed_cgm - true_bg,
            }
        )

        action = np.array(
            [0, 0],
            dtype=np.int64,
        )

        (
            obs,
            reward,
            cost,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        step_idx += 1

        done = terminated or truncated

    # final observation
    true_bg, cgm_hist, observed_cgm = get_current_values(
        env,
        obs,
    )

    records.append(
        {
            "step": step_idx,
            "time_min": step_idx * SAMPLE_TIME,
            "time_hour": step_idx * SAMPLE_TIME / 60,
            "true_bg": true_bg,
            "cgm": cgm_hist,
            "obs_cgm": observed_cgm,
            "sensor_error": observed_cgm - true_bg,
        }
    )

    return pd.DataFrame(records)


# ==========================================
# Nominal
# ==========================================

nominal_env = make_env(NOMINAL_SIGMA)

nominal_df = rollout(
    nominal_env,
    SEED,
)

nominal_scenario = np.asarray(
    nominal_env.unwrapped.scenario_meals
)

print("\n===== Nominal meals =====")
nominal_env.unwrapped._print_scenario_meals(
    nominal_env.unwrapped.scenario_meals
)

nominal_env.close()


# ==========================================
# Sensor stress
# ==========================================

stress_env = make_env(STRESS_SIGMA)

stress_df = rollout(
    stress_env,
    SEED,
)

stress_scenario = np.asarray(
    stress_env.unwrapped.scenario_meals
)

print("\n===== Stress meals =====")
stress_env.unwrapped._print_scenario_meals(
    stress_env.unwrapped.scenario_meals
)

stress_env.close()


# ==========================================
# Sanity checks
# ==========================================

print("\n===== Sanity checks =====")

print(
    "Meal scenarios identical:",
    np.allclose(
        nominal_scenario,
        stress_scenario,
    ),
)

true_bg_diff = np.abs(
    nominal_df["true_bg"].to_numpy()
    - stress_df["true_bg"].to_numpy()
)

print(
    "Max true BG difference:",
    true_bg_diff.max(),
)

nominal_error = (
    nominal_df["obs_cgm"]
    - nominal_df["true_bg"]
)

stress_error = (
    stress_df["obs_cgm"]
    - stress_df["true_bg"]
)

print(
    "Nominal sensor error std:",
    nominal_error.std(),
)

print(
    "Stress sensor error std:",
    stress_error.std(),
)


# ==========================================
# Save CSV
# ==========================================

nominal_df.to_csv(
    OUT_DIR / "nominal_sensor.csv",
    index=False,
)

stress_df.to_csv(
    OUT_DIR / "sensor_sigma_15.csv",
    index=False,
)


# ==========================================
# Plot
# ==========================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 9),
    sharex=True,
)


# BG / CGM comparison
axes[0].plot(
    nominal_df["time_hour"],
    nominal_df["true_bg"],
    label="True BG",
    linewidth=2.5,
)

axes[0].plot(
    nominal_df["time_hour"],
    nominal_df["obs_cgm"],
    label=f"Nominal CGM (σ={NOMINAL_SIGMA})",
    linewidth=1.2,
    alpha=0.8,
)

axes[0].plot(
    stress_df["time_hour"],
    stress_df["obs_cgm"],
    label=f"Stressed CGM (σ={STRESS_SIGMA})",
    linewidth=1.2,
    alpha=0.8,
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

axes[0].set_ylabel("Glucose (mg/dL)")
axes[0].set_title(
    f"{PATIENT}: CGM sensor noise stress"
)

axes[0].grid(alpha=0.3)
axes[0].legend()


# Sensor error
axes[1].plot(
    nominal_df["time_hour"],
    nominal_error,
    label=f"Nominal error (σ={NOMINAL_SIGMA})",
    linewidth=1,
)

axes[1].plot(
    stress_df["time_hour"],
    stress_error,
    label=f"Stress error (σ={STRESS_SIGMA})",
    linewidth=1,
)

axes[1].axhline(
    0,
    linestyle="--",
    linewidth=1,
)

axes[1].set_xlabel("Time (hours)")
axes[1].set_ylabel("CGM - true BG (mg/dL)")
axes[1].set_xlim(0, 24)

axes[1].grid(alpha=0.3)
axes[1].legend()


plt.tight_layout()

plot_path = OUT_DIR / "nominal_vs_sensor_noise.png"

plt.savefig(
    plot_path,
    dpi=150,
)

plt.show()

print("\nSaved:", plot_path)