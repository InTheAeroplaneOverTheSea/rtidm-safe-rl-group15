# Author-v1 Shield stress tests on SURF

This workflow evaluates all eight January-2026 T1D policies with and without
the predictive Shield while keeping the author-v1 GlucoAlg, GlucoSim,
checkpoints, transition data, and BA-NODE predictor pinned.

## 1. Scenarios

The stress evaluator supports three paired scenarios:

| name | intervention |
| --- | --- |
| `nominal` | no additional stress; use this as the matched control |
| `sensor_sigma15` | set CGM observation-noise sigma to 15 mg/dL |
| `unannounced_meal_60g` | inject 60 g at 15:00 on every simulated day |

The unannounced meal is deliberately different from an announced schedule
shift. At 15:00 the policy and Shield first select an action from the current
observation. Only then does the evaluator add 60 g to the real patient's pending
carbohydrate buffer, immediately before `env.step`. The meal is never inserted
into `scenario_meals`, so `time_until_meal`, `next_meal_size`, and the pre-bolus
flag cannot reveal it before the event. Subsequent COB and CGM observations do
show its physiological consequences.

The old announced `meal_shift_minutes` API remains available for simulator
experiments, but it is not used by this paper-aligned stress workflow.

## 2. Reproducible pairing

January GlucoSim constructs its JAX key from zero and does not reliably honor a
later `reset(seed=...)`. It also advances that key on every environment step.
Consequently, two multi-episode controllers can receive different later meal
schedules if one terminates earlier.

The stress adapter fixes evaluation seeding without modifying the pinned
simulator archive. Before every episode it sets a fresh raw JAX key derived as:

```text
SeedSequence([base_seed=22, replicate_id, episode, 20261011])
```

The condition and algorithm are intentionally absent. Matching no-Shield and
Shield cells therefore receive the same episode seeds, normal meal schedule,
sensor-noise stream, and hidden-meal plan regardless of earlier termination.
Every cell records these values in `stress_episode_metadata.json`.

## 3. Prerequisites

Complete the checkpoint and transition-download steps in
`docs/AUTHOR_V1_T1D_VALIDATION_TUTORIAL.md` first. From the repository root on
SURF, verify:

```bash
test -f author_v1_GlucoAlg_50d3134.tar.gz
test -f author_v1_GlucoSim_6522d49.tar.gz
test -f author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json
test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth
test -d external/GlucoAlg/saved_files/env_transitions
```

The Slurm job verifies the old source archives, January checkpoint revisions,
transition revision, policy hashes, and predictor hash before evaluating.

## 4. Pilot one algorithm first

Start with CPO, one episode per patient/checkpoint cell, replicate zero:

```bash
bash scripts/submit_author_v1_stress_pair.sh \
  CPO sensor_sigma15 1 0 10

bash scripts/submit_author_v1_stress_pair.sh \
  CPO unannounced_meal_60g 1 0 10
```

Each command submits two array-cell-zero smoke jobs. The remaining 80 cells for
each condition start only after their corresponding smoke job succeeds.

For a matched unstressed control under the corrected seed handling, also run:

```bash
bash scripts/submit_author_v1_stress_pair.sh \
  CPO nominal 1 0 10
```

Do not compare the stress runs numerically against an older nominal collection
that used the stateful fixed-zero key behavior.

## 5. Submit all eight algorithms

After the CPO smoke/full arrays and metadata validation pass:

```bash
bash scripts/submit_author_v1_stress_matrix.sh sensor_sigma15 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh unannounced_meal_60g 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh nominal 1 0 10
```

Each command submits 81 no-Shield cells and 81 predictive-Shield cells for each
of eight algorithms. One stress scenario therefore contains 1,296 array tasks.

Results are written under:

```text
author_v1_stress_evaluation/
  <scenario>/
    replicate0/
      no_shield/t1d/<algorithm>/...
      predictive_penalty10/t1d/<algorithm>/...
```

Use a different replicate id to obtain a new, still exactly paired environment
randomization. Reusing an id intentionally reproduces the same disturbances.

## 6. Validate and aggregate

Example for the CPO sensor run:

```bash
ROOT=author_v1_stress_evaluation/sensor_sigma15/replicate0

python scripts/aggregate_author_v1_t1d.py \
  "$ROOT/no_shield/t1d/CPO" \
  --expected-episodes 1 \
  --algorithm CPO \
  --condition no_shield \
  --stress-scenario sensor_sigma15 \
  --replicate-id 0 \
  --expected-result-collection author_v1_stress_evaluation

python scripts/aggregate_author_v1_t1d.py \
  "$ROOT/predictive_penalty10/t1d/CPO" \
  --expected-episodes 1 \
  --algorithm CPO \
  --condition predictive \
  --logit-penalty 10 \
  --stress-scenario sensor_sigma15 \
  --replicate-id 0 \
  --expected-result-collection author_v1_stress_evaluation

python scripts/validate_author_v1_stress_pairing.py \
  "$ROOT/no_shield/t1d/CPO" \
  "$ROOT/predictive_penalty10/t1d/CPO" \
  --expected-episodes 1 \
  --logit-penalty 10
```

Repeat with `unannounced_meal_60g` and `nominal`. Do not report a condition
unless both aggregation validation and pairing validation print `PASS`.

For unannounced meals, fewer delivered events than planned is valid when an
episode terminates early. The paired validator compares the seed and planned
disturbance, not the number of events reached by the two controllers.

## 7. Recommended analysis

For every algorithm report at least:

1. Shield minus no-Shield under the same stress;
2. stressed minus nominal for the same Shield condition;
3. early-termination count;
4. TIR, Risk Index, and CV;
5. the fraction of cells in which Shield worsens each clinical metric.

A Shield "break" should not be defined only as lower mean TIR. Also flag cases
where the predictive Shield increases early termination, Risk Index, or glucose
variability relative to its exactly paired no-Shield controller.
