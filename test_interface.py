import numpy as np

from interface import GlucoSimInterface, StressConfig


PATIENT = "adult#001"
SEED = 0


interface = GlucoSimInterface(
    patient_name=PATIENT,
    simulation_minutes=1440,
    sample_time=5,
)


def rollout(env, obs):
    true_bg = []
    cgm = []
    cob = []

    while True:
        e = env.unwrapped
        idx = int(e._jax_state["index"]) - 1

        true_bg.append(
            float(e._jax_state["BG_hist"][idx])
        )

        cgm.append(
            float(obs[0])
        )

        cob.append(
            float(obs[2])
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
        "cob": np.asarray(cob),
    }


# ============================================================
# 1. Nominal
# ============================================================

print("\n==============================")
print("TEST 1: NOMINAL")
print("==============================")

env, obs, info = interface.create(
    stress=StressConfig(),
    seed=SEED,
)

print("Initial obs:", obs)

print("Patient kabs:",
      env.unwrapped.get_patient_params().kabs)

print("CGM sigma:",
      env.unwrapped.env_params.noise_config.cgm_obs_sigma_mgdl)

print("Meal scenario:")
env.unwrapped._print_scenario_meals(
    env.unwrapped.scenario_meals
)

nominal_meals = np.asarray(
    env.unwrapped.scenario_meals
).copy()

nominal = rollout(
    env,
    obs,
)

env.close()

print("Nominal max true BG:",
      nominal["true_bg"].max())


# ============================================================
# 2. Meal timing stress
# ============================================================

print("\n==============================")
print("TEST 2: MEAL SHIFT +60 MIN")
print("==============================")

env, obs, info = interface.create(
    stress=StressConfig(
        meal_shift_minutes=60,
    ),
    seed=SEED,
)

print("Shifted meal scenario:")
env.unwrapped._print_scenario_meals(
    env.unwrapped.scenario_meals
)

shifted_meals = np.asarray(
    env.unwrapped.scenario_meals
).copy()

meal_shift = rollout(
    env,
    obs,
)

env.close()


valid = nominal_meals[..., 0] >= 0

meal_time_difference = (
    shifted_meals[..., 0][valid]
    - nominal_meals[..., 0][valid]
)

print(
    "Meal timing differences:",
    meal_time_difference,
)

print(
    "Meal amounts identical:",
    np.allclose(
        nominal_meals[..., 1],
        shifted_meals[..., 1],
    ),
)


# ============================================================
# 3. Sensor noise stress
# ============================================================

print("\n==============================")
print("TEST 3: CGM NOISE")
print("==============================")

env, obs, info = interface.create(
    stress=StressConfig(
        cgm_noise_sigma=15.0,
    ),
    seed=SEED,
)

print(
    "Stress CGM sigma:",
    env.unwrapped.env_params.noise_config.cgm_obs_sigma_mgdl,
)

sensor = rollout(
    env,
    obs,
)

env.close()


true_bg_diff = np.abs(
    nominal["true_bg"]
    - sensor["true_bg"]
)

nominal_sensor_error = (
    nominal["cgm"]
    - nominal["true_bg"]
)

stress_sensor_error = (
    sensor["cgm"]
    - sensor["true_bg"]
)

print(
    "Max true BG difference:",
    true_bg_diff.max(),
)

print(
    "Nominal sensor error std:",
    nominal_sensor_error.std(),
)

print(
    "Stress sensor error std:",
    stress_sensor_error.std(),
)


# ============================================================
# 4. kabs OOD
# ============================================================

print("\n==============================")
print("TEST 4: KABS OOD")
print("==============================")

env, obs, info = interface.create(
    stress=StressConfig(
        kabs_scale=1.5,
    ),
    seed=SEED,
)

print(
    "OOD kabs:",
    env.unwrapped.get_patient_params().kabs,
)

ood_meals = np.asarray(
    env.unwrapped.scenario_meals
).copy()

ood = rollout(
    env,
    obs,
)

env.close()


ood_diff = np.abs(
    nominal["true_bg"]
    - ood["true_bg"]
)

print(
    "Meal scenarios identical:",
    np.allclose(
        nominal_meals,
        ood_meals,
    ),
)

print(
    "Max true BG difference:",
    ood_diff.max(),
)

print(
    "Mean true BG difference:",
    ood_diff.mean(),
)

print("\n==============================")
print("TEST 5: COMBINED STRESS")
print("==============================")

env, obs, info = interface.create(
    stress=StressConfig(
        meal_shift_minutes=60,
        cgm_noise_sigma=15.0,
        kabs_scale=1.5,
    ),
    seed=SEED,
)

print(
    "kabs:",
    env.unwrapped.get_patient_params().kabs,
)

print(
    "CGM sigma:",
    env.unwrapped.env_params.noise_config.cgm_obs_sigma_mgdl,
)

print("Meal scenario:")
env.unwrapped._print_scenario_meals(
    env.unwrapped.scenario_meals
)

combined = rollout(env, obs)

env.close()
# ============================================================
# Summary
# ============================================================

print("\n==============================")
print("INTERFACE TEST SUMMARY")
print("==============================")

meal_ok = np.allclose(
    meal_time_difference,
    60,
)

sensor_ok = (
    true_bg_diff.max() < 1.0
    and stress_sensor_error.std()
    > nominal_sensor_error.std()
)

ood_ok = (
    ood_diff.max() > 1.0
)

print("Meal shift working:", meal_ok)
print("Sensor stress working:", sensor_ok)
print("OOD stress working:", ood_ok)

if meal_ok and sensor_ok and ood_ok:
    print("\nALL INTERFACE TESTS PASSED")
else:
    print("\nSOME INTERFACE TESTS FAILED")