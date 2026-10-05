interface.py is where we can modify the simulator. There're three factors: noise, meals, ood. 

```markdown
# GlucoSim Stress Interface

This interface provides a unified way to create GlucoSim environments under
nominal and stressed conditions.

It currently supports three types of simulation stress:

- Meal timing shift
- CGM sensor noise
- Physiological OOD through carbohydrate absorption dynamics (`kabs`)

The goal is to keep all simulator-specific stress manipulation inside
`interface.py`, so evaluation code and RL policies do not need to directly
modify GlucoSim internals.

---

## 1. Project Structure

Example project structure:

```text
xxxxx/
├── GlucoSim/
├── GlucoAlg/
│
├── interface.py
├── test_interface.py
│
├── experiments/
│   └── ...
│
└── outputs/
```

`interface.py` should be treated as the main entry point for creating
simulation environments.

Evaluation scripts should avoid directly modifying:

```python
env.unwrapped._jax_state
env.unwrapped.env_params
scenario_meals
NoiseConfig
patient_overrides
```

These implementation details are handled by `GlucoSimInterface`.

---

## 2. Basic API

Import the interface:

```python
from interface import GlucoSimInterface, StressConfig
```

Create an interface instance:

```python
interface = GlucoSimInterface(
    patient_name="adult#001",
    simulation_minutes=1440,
    sample_time=5,
)
```

Then create an environment:

```python
env, obs, info = interface.create(
    stress=StressConfig(),
    seed=0,
)
```

The returned objects are:

```text
env   : configured GlucoSim Gym environment
obs   : initial observation after reset
info  : initial Gym information dictionary
```

---

## 3. StressConfig

The current configuration contains three stress factors:

```python
StressConfig(
    meal_shift_minutes=0.0,
    cgm_noise_sigma=None,
    kabs_scale=1.0,
)
```

Default values correspond to the nominal simulator condition.

### Parameters

#### `meal_shift_minutes`

Shifts the timing of all automatically generated meals.

Default:

```python
meal_shift_minutes = 0.0
```

Example:

```python
StressConfig(
    meal_shift_minutes=60,
)
```

A nominal schedule such as:

```text
05:47  43 g
12:39  79 g
18:28  80 g
21:42  11 g
```

becomes:

```text
06:47  43 g
13:39  79 g
19:28  80 g
22:42  11 g
```

Meal amounts are unchanged.

This stress modifies:

```text
meal timing
    ↓
carbohydrate intake
    ↓
patient physiology
    ↓
blood glucose
```

---

#### `cgm_noise_sigma`

Changes CGM observation noise.

Default:

```python
cgm_noise_sigma = None
```

`None` means that the simulator's default noise configuration is used.

For the current GlucoSim version, the nominal value is:

```text
cgm_obs_sigma_mgdl = 1.5 mg/dL
```

Example:

```python
StressConfig(
    cgm_noise_sigma=15.0,
)
```

This modifies the observation seen by the agent:

```text
true BG
   ↓
CGM sensor model
   ↓
sensor noise
   ↓
obs[0]
```

It does not directly modify patient physiology.

In our sanity test:

```text
Maximum true BG difference = 0.0 mg/dL
```

while CGM measurement error increased substantially.

---

#### `kabs_scale`

Changes the patient's carbohydrate absorption dynamics.

Default:

```python
kabs_scale = 1.0
```

Example:

```python
StressConfig(
    kabs_scale=1.5,
)
```

The patient's original `kabs` is multiplied by the requested scale:

```text
new kabs = nominal kabs × kabs_scale
```

For `adult#001`:

```text
nominal kabs = 0.08906
```

Therefore:

```text
kabs_scale = 1.5
new kabs   = 0.13359
```

This represents a physiological / dynamics OOD condition.

Unlike sensor noise, this changes the underlying true blood glucose dynamics.

---

## 4. Nominal Environment

To create the nominal simulator:

```python
from interface import GlucoSimInterface, StressConfig

interface = GlucoSimInterface(
    patient_name="adult#001",
)

env, obs, info = interface.create(
    stress=StressConfig(),
    seed=0,
)
```

Equivalent configuration:

```python
StressConfig(
    meal_shift_minutes=0,
    cgm_noise_sigma=None,
    kabs_scale=1.0,
)
```

---

## 5. Meal Timing Stress

Example: shift every meal by 60 minutes.

```python
stress = StressConfig(
    meal_shift_minutes=60,
)

env, obs, info = interface.create(
    stress=stress,
    seed=0,
)
```

To inspect the resulting meal schedule:

```python
env.unwrapped._print_scenario_meals(
    env.unwrapped.scenario_meals
)
```

Example output:

```text
Generated meal scenario:
  Day 1: 06:47: 43g | 13:39: 79g | 19:28: 80g | 22:42: 11g
```

For controlled experiments, use the same seed for nominal and stressed
environments.

---

## 6. Sensor Noise Stress

Example:

```python
stress = StressConfig(
    cgm_noise_sigma=15.0,
)

env, obs, info = interface.create(
    stress=stress,
    seed=0,
)
```

The current CGM noise setting can be inspected with:

```python
print(
    env.unwrapped
    .env_params
    .noise_config
    .cgm_obs_sigma_mgdl
)
```

Expected output:

```text
15.0
```

The policy observes CGM through:

```python
obs[0]
```

while the simulator keeps true physiological BG separately.

The current true BG can be obtained for debugging with:

```python
e = env.unwrapped

idx = int(e._jax_state["index"]) - 1

true_bg = float(
    e._jax_state["BG_hist"][idx]
)

observed_cgm = float(obs[0])
```

This direct access should mainly be used for evaluation and visualization,
not by the policy.

---

## 7. Physiological OOD

Example: faster carbohydrate absorption.

```python
stress = StressConfig(
    kabs_scale=1.5,
)

env, obs, info = interface.create(
    stress=stress,
    seed=0,
)
```

Check the resulting patient parameter:

```python
print(
    env.unwrapped
    .get_patient_params()
    .kabs
)
```

For `adult#001`, the expected value is approximately:

```text
0.13359
```

This stress changes underlying patient dynamics and therefore changes true BG.

---

## 8. Combined Stress

Multiple stress factors can be enabled simultaneously.

```python
stress = StressConfig(
    meal_shift_minutes=60,
    cgm_noise_sigma=15.0,
    kabs_scale=1.5,
)

env, obs, info = interface.create(
    stress=stress,
    seed=0,
)
```

The interface automatically applies the factors in the required order:

```text
patient dynamics override
        ↓
create environment
        ↓
configure sensor noise
        ↓
reset environment
        ↓
generate meal scenario
        ↓
apply meal timing shift
        ↓
return configured environment
```

This ordering is important because the three stress factors enter the
simulator at different stages.

---

## 9. Running a Rollout

Once the environment is created, it behaves like the original GlucoSim
environment.

Example with no agent intervention:

```python
import numpy as np

env, obs, info = interface.create(
    stress=StressConfig(),
    seed=0,
)

done = False

while not done:

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

    done = terminated or truncated

env.close()
```

For this GlucoSim version:

```text
action[0] = bolus level
action[1] = agent-requested meal level
```

Both use discrete levels:

```text
0.00
0.25
0.50
0.75
1.00
```

Therefore:

```python
action = [0, 0]
```

means no agent-requested bolus and no additional agent-requested meal.

It does not disable the simulator's automatically generated meal scenario.

---

## 10. Using the Interface with a Policy

The interface is intentionally independent of the controller.

A future CPO evaluation can therefore use:

```python
interface = GlucoSimInterface(
    patient_name="adult#001",
)

env, obs, info = interface.create(
    stress=StressConfig(
        meal_shift_minutes=60,
    ),
    seed=0,
)

done = False

while not done:

    action = policy(obs)

    (
        obs,
        reward,
        cost,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    done = terminated or truncated

env.close()
```

The same controller can then be evaluated under different stress conditions
without modifying the policy code.

---

## 11. Recommended Experiment Conditions

The current interface supports a simple experiment matrix such as:

```text
Condition        meal shift    CGM sigma    kabs scale
-------------------------------------------------------
Nominal              0          default        1.0
Meal stress         60          default        1.0
Sensor stress        0           15.0          1.0
Dynamics OOD         0          default        1.5
Combined            60           15.0          1.5
```

The exact severity values are currently experimental and should not yet be
treated as final benchmark settings.

---

## 12. Reproducibility

For paired comparisons, always use the same seed.

For example:

```python
nominal_env, nominal_obs, _ = interface.create(
    stress=StressConfig(),
    seed=0,
)

stress_env, stress_obs, _ = interface.create(
    stress=StressConfig(
        meal_shift_minutes=60,
    ),
    seed=0,
)
```

Using fresh environments with the same seed ensures that the generated meal
scenario and other stochastic components are aligned as closely as possible.

Do not reuse the same environment object for nominal and stressed experiments.

---

## 13. Interface Validation

The interface can be tested with:

```powershell
uv run python test_interface.py
```

A successful test currently produces:

```text
Meal shift working: True
Sensor stress working: True
OOD stress working: True

ALL INTERFACE TESTS PASSED
```

The existing tests verify that:

```text
Meal shift:
meal times change by exactly the requested amount
meal amounts remain identical

Sensor stress:
true BG remains unchanged
CGM measurement error increases

kabs OOD:
meal scenario remains unchanged
true BG dynamics change
```

---

## 14. Current Validated Configuration

The interface has currently been validated with:

```text
Environment:
t1d-v0

Patient:
adult#001

Simulation duration:
1440 minutes

Sample time:
5 minutes

Seed:
0
```

Validated stress examples:

```text
Meal timing:
+60 minutes

CGM observation noise:
σ = 15 mg/dL

Carbohydrate absorption OOD:
kabs × 1.5
```

These values were selected for sanity testing and visualization.

They are not yet final benchmark severity levels.

---

## 15. Design Principle

The main design principle is:

```text
Controller / evaluation code
          │
          │ only sees:
          │
          ▼
GlucoSimInterface.create(...)
          │
          ▼
configured GlucoSim environment
```

Simulator implementation details remain inside `interface.py`.

This makes it possible to evaluate:

```text
CPO
CPO + Shield
```

under identical:

```text
Nominal
Meal shift
Sensor noise
Dynamics OOD
```

conditions without changing the controller implementation.
```
