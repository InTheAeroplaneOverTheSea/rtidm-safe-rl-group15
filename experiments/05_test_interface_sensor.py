import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interface import GlucoSimInterface, StressConfig

PATIENT = "adult#001"
SEED = 0
SIMULATION_MINUTES = 1440
SAMPLE_TIME = 5


interface = GlucoSimInterface(
    patient_name=PATIENT,
    simulation_minutes=SIMULATION_MINUTES,
    sample_time=SAMPLE_TIME,
)


def rollout(env, obs):
    true_bg = []
    cgm = []

    while True:
        e = env.unwrapped
        idx = int(e._jax_state["index"]) - 1

        true_bg.append(
            float(e._jax_state["BG_hist"][idx])
        )

        cgm.append(
            float(obs[0])
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

        if terminated or truncated:
            break

    return {
        "true_bg": np.asarray(true_bg),
        "cgm": np.asarray(cgm),
    }


def run_condition(name, config):
    print("\n==============================")
    print(name)
    print("==============================")

    env, obs, info = interface.create(
        stress=config,
        seed=SEED,
    )

    noise = env.unwrapped.env_params.noise_config

    print("cgm_obs_sigma_mgdl:",
          noise.cgm_obs_sigma_mgdl)

    print("cgm_bias_mgdl:",
          noise.cgm_bias_mgdl)

    print("cgm_rw_sigma_bias:",
          noise.cgm_rw_sigma_bias)

    print("cgm_dropout_prob:",
          noise.cgm_dropout_prob)

    print("use_ou:",
          noise.use_ou)

    print("ou_sigma_dL:",
          noise.ou_sigma_dL)

    result = rollout(env, obs)

    env.close()

    return result


# ============================================================
# Nominal
# ============================================================

nominal = run_condition(
    "NOMINAL",
    StressConfig(),
)


# ============================================================
# Constant bias
# ============================================================

bias = run_condition(
    "BIAS +15 mg/dL",
    StressConfig(
        cgm_bias_mgdl=15.0,
    ),
)


# ============================================================
# Random-walk drift
# ============================================================

drift = run_condition(
    "DRIFT",
    StressConfig(
        cgm_rw_sigma_bias=0.02,
    ),
)


# ============================================================
# Dropout
# ============================================================

dropout = run_condition(
    "DROPOUT 10%",
    StressConfig(
        cgm_dropout_prob=0.10,
    ),
)


# ============================================================
# OU correlated noise
# ============================================================

ou = run_condition(
    "OU NOISE",
    StressConfig(
        cgm_use_ou=True,
        cgm_ou_sigma=10.0,
    ),
)


# ============================================================
# Analysis
# ============================================================

conditions = {
    "Bias": bias,
    "Drift": drift,
    "Dropout": dropout,
    "OU": ou,
}


print("\n======================================")
print("TRUE BG SANITY CHECK")
print("======================================")

for name, result in conditions.items():

    diff = np.abs(
        nominal["true_bg"]
        - result["true_bg"]
    )

    print(
        f"{name:10s}",
        "max true BG diff =",
        np.nanmax(diff),
    )


print("\n======================================")
print("SENSOR ERROR")
print("======================================")

nominal_error = (
    nominal["cgm"]
    - nominal["true_bg"]
)

print(
    "Nominal mean error:",
    np.nanmean(nominal_error),
)

print(
    "Nominal error std:",
    np.nanstd(nominal_error),
)


for name, result in conditions.items():

    error = (
        result["cgm"]
        - result["true_bg"]
    )

    print(f"\n{name}")

    print(
        "mean error:",
        np.nanmean(error),
    )

    print(
        "error std:",
        np.nanstd(error),
    )

    print(
        "min error:",
        np.nanmin(error),
    )

    print(
        "max error:",
        np.nanmax(error),
    )


# ============================================================
# Inspect dropout behavior
# ============================================================

dropout_cgm = dropout["cgm"]

print("\n======================================")
print("DROPOUT INSPECTION")
print("======================================")

print(
    "NaN count:",
    np.isnan(dropout_cgm).sum(),
)

print(
    "Zero count:",
    np.sum(dropout_cgm == 0),
)

repeated = np.isclose(
    dropout_cgm[1:],
    dropout_cgm[:-1],
)

print(
    "Repeated consecutive CGM values:",
    repeated.sum(),
)

print(
    "Total transitions:",
    len(dropout_cgm) - 1,
)


# ============================================================
# Plot
# ============================================================

time = (
    np.arange(len(nominal["cgm"]))
    * SAMPLE_TIME
    / 60
)

fig, axes = plt.subplots(
    5,
    1,
    figsize=(13, 15),
    sharex=True,
)


plot_data = [
    ("Nominal", nominal),
    ("Bias +15 mg/dL", bias),
    ("Drift", drift),
    ("Dropout 10%", dropout),
    ("OU noise", ou),
]


for ax, (name, result) in zip(
    axes,
    plot_data,
):
    ax.plot(
        time,
        result["true_bg"],
        label="True BG",
        linewidth=2,
    )

    ax.plot(
        time,
        result["cgm"],
        label="CGM",
        linewidth=1,
        alpha=0.8,
    )

    ax.set_ylabel("mg/dL")
    ax.set_title(name)

    ax.grid(alpha=0.3)
    ax.legend()


axes[-1].set_xlabel("Time (hours)")
axes[-1].set_xlim(0, 24)

plt.tight_layout()
plt.show()