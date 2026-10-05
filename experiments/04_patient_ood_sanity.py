import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glucosim
from glucosim import gym_env as gym


PATIENT = "adult#001"
SEED = 0
SIMULATION_MINUTES = 1440
SAMPLE_TIME = 5
BASE_VMX = 0.008146460500252993
BASE_KABS = 0.08906


def make_env(kabs_scale):
    return gym.make(
        "t1d-v0",
        simulation_minutes=SIMULATION_MINUTES,
        sample_time=SAMPLE_TIME,
        patient_name=PATIENT,
        patient_overrides={
            "kabs": BASE_KABS * kabs_scale,
        },
    )

def make_env_02(vmx_scale):
    return gym.make(
        "t1d-v0",
        simulation_minutes=SIMULATION_MINUTES,
        sample_time=SAMPLE_TIME,
        patient_name=PATIENT,
        patient_overrides={
            "Vmx": BASE_VMX * vmx_scale,
        },
    )



def make_env_01(ir_factor):
    return gym.make(
        "t1d-v0",
        simulation_minutes=SIMULATION_MINUTES,
        sample_time=SAMPLE_TIME,
        patient_name=PATIENT,
        patient_overrides={
            "insulin_resistance_factor": ir_factor,
        },
    )


def get_true_bg(env):
    e = env.unwrapped
    idx = int(e._jax_state["index"]) - 1
    return float(e._jax_state["BG_hist"][idx])


def rollout(env):
    obs, _ = env.reset(seed=SEED)

    records = []
    step_idx = 0

    while True:
        records.append({
            "time_hour": step_idx * SAMPLE_TIME / 60,
            "true_bg": get_true_bg(env),
            "cgm": float(obs[0]),
        })

        action = np.array([0, 0], dtype=np.int64)

        obs, reward, cost, terminated, truncated, info = env.step(action)

        step_idx += 1

        if terminated or truncated:
            break

    records.append({
        "time_hour": step_idx * SAMPLE_TIME / 60,
        "true_bg": get_true_bg(env),
        "cgm": float(obs[0]),
    })

    return pd.DataFrame(records)


# ---------------------
# Nominal
# ---------------------

nominal_env = make_env(1.0)
nominal_df = rollout(nominal_env)

nominal_meals = np.asarray(
    nominal_env.unwrapped.scenario_meals
)

print("Nominal IR:",
      nominal_env.unwrapped.get_patient_params().insulin_resistance_factor)

nominal_env.close()


# ---------------------
# OOD
# ---------------------
slow_env = make_env(0.5)
fast_env = make_env(1.5)
slow_df = rollout(slow_env)
fast_df = rollout(fast_env)

slow_meals =  np.asarray(
    slow_env.unwrapped.scenario_meals
)
fast_meals = np. asarray(
    fast_env.unwrapped.scenario_meals
)

print("FAST IR:",
      fast_env.unwrapped.get_patient_params().insulin_resistance_factor)
print("SLOW IR:",
      slow_env.unwrapped.get_patient_params().insulin_resistance_factor)

slow_env.close()
fast_env.close()

ood_env = make_env(0.5)
ood_df = rollout(ood_env)

ood_meals = np.asarray(
    ood_env.unwrapped.scenario_meals
)

print("OOD IR:",
      ood_env.unwrapped.get_patient_params().insulin_resistance_factor)

ood_env.close()


# ---------------------
# Checks
# ---------------------

print(
    "Meal scenarios identical:",
    np.allclose(nominal_meals, ood_meals),
)

diff = np.abs(
    nominal_df["true_bg"].to_numpy()
    - ood_df["true_bg"].to_numpy()
)

print("Max true BG difference:", diff.max())
print("Mean true BG difference:", diff.mean())

print("Nominal max BG:", nominal_df["true_bg"].max())
print("OOD max BG:", ood_df["true_bg"].max())
print(
    "Nominal Vmx:",
    nominal_env.unwrapped.get_patient_params().Vmx,
)

print(
    "OOD Vmx:",
    ood_env.unwrapped.get_patient_params().Vmx,
)

# ---------------------
# Plot
# ---------------------

plt.figure(figsize=(12, 5))

plt.plot(
    nominal_df["time_hour"],
    nominal_df["true_bg"],
    label="Nominal IR = 1.0",
    linewidth=2,
)

plt.plot(
    ood_df["time_hour"],
    ood_df["true_bg"],
    label="OOD IR = 1.5",
    linewidth=2,
)
plt.plot(
    nominal_df["time_hour"],
    nominal_df["true_bg"],
    label="Nominal kabs ×1.0",
)

plt.plot(
    slow_df["time_hour"],
    slow_df["true_bg"],
    label="Slow absorption kabs ×0.5",
)

plt.plot(
    fast_df["time_hour"],
    fast_df["true_bg"],
    label="Fast absorption kabs ×1.5",
)
plt.axhline(70, linestyle="--", linewidth=1)
plt.axhline(180, linestyle="--", linewidth=1)

plt.xlabel("Time (hours)")
plt.ylabel("True BG (mg/dL)")
plt.title("adult#001: insulin-resistance OOD sanity check")

plt.xlim(0, 24)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()