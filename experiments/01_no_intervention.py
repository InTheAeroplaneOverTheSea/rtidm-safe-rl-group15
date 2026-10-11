from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glucosim
from glucosim import gym_env as gym


# =========================
# Config
# =========================
PATIENT = "adult#001"
SIMULATION_MINUTES = 24 * 60
SAMPLE_TIME = 5
SEED = 0

OUT_DIR = Path("outputs/no_intervention")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Build env
# =========================
env = gym.make(
    "t1d-v0",
    simulation_minutes=SIMULATION_MINUTES,
    sample_time=SAMPLE_TIME,
    patient_name=PATIENT,
)

obs, info = env.reset(seed=SEED)

print("Initial obs:", obs)
print("Initial info:", info)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# =========================
# Rollout
# =========================
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

    # strict no intervention
    action = np.array([0, 0], dtype=np.int64)

    obs, reward, cost, terminated, truncated, info = env.step(action)

    step_idx += 1
    done = terminated or truncated

    if done:
        # also store final state after last step
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

env.close()


# =========================
# Save table
# =========================
df = pd.DataFrame(records)
csv_path = OUT_DIR / f"{PATIENT}_no_intervention.csv"
df.to_csv(csv_path, index=False)

print("\n===== Summary =====")
print("Number of rows:", len(df))
print("Initial BG:", df["bg"].iloc[0])
print("Final BG:", df["bg"].iloc[-1])
print("Min BG:", df["bg"].min())
print("Max BG:", df["bg"].max())
print("Saved CSV to:", csv_path)


# =========================
# Plot 1: BG only
# =========================
plt.figure(figsize=(12, 5))
plt.plot(df["time_hour"], df["bg"], linewidth=2, label="BG")
plt.axhline(70, linestyle="--", linewidth=1, label="70 mg/dL")
plt.axhline(180, linestyle="--", linewidth=1, label="180 mg/dL")
plt.xlabel("Time (hours)")
plt.ylabel("Blood glucose (mg/dL)")
plt.title(f"{PATIENT} - no intervention")
plt.xlim(0, 24)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

bg_png = OUT_DIR / f"{PATIENT}_bg_curve.png"
plt.savefig(bg_png, dpi=150)
plt.show()

print("Saved BG plot to:", bg_png)


# =========================
# Plot 2: BG / IOB / COB
# =========================
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df["time_hour"], df["bg"], linewidth=2)
plt.axhline(70, linestyle="--", linewidth=1)
plt.axhline(180, linestyle="--", linewidth=1)
plt.ylabel("BG")
plt.xlim(0, 24)
plt.grid(alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(df["time_hour"], df["iob"], linewidth=2)
plt.ylabel("IOB")
plt.xlim(0, 24)
plt.grid(alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(df["time_hour"], df["cob"], linewidth=2)
plt.xlabel("Time (hours)")
plt.ylabel("COB")
plt.xlim(0, 24)
plt.grid(alpha=0.3)

plt.suptitle(f"{PATIENT} - no intervention rollout")
plt.tight_layout()

full_png = OUT_DIR / f"{PATIENT}_bg_iob_cob.png"
plt.savefig(full_png, dpi=150)
plt.show()

print("Saved BG/IOB/COB plot to:", full_png)