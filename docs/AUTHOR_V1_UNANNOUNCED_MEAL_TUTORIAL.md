# Author-v1 unannounced-meal Shield tutorial

Last updated: 2026-10-11

The submitted CPO run and its one-time initial Slurm snapshot are recorded in
[`AUTHOR_V1_UNANNOUNCED_MEAL_RUN_STATUS.md`](AUTHOR_V1_UNANNOUNCED_MEAL_RUN_STATUS.md).

This tutorial is only for the paper-aligned unexpected-meal experiment. Sensor
noise, bias, drift, and dropout are documented separately in
[`AUTHOR_V1_STRESS_TEST_TUTORIAL.md`](AUTHOR_V1_STRESS_TEST_TUTORIAL.md).

The first comparison is one matched CPO cell under the same hidden meal:

| Arm | Controller | Unexpected meal | Shield meal-hyper check |
| --- | --- | --- | --- |
| A | no Shield | `60 g` daily at `15:00` | not applicable |
| B | predictive Shield, penalty `10` | `60 g` daily at `15:00` | `false` |

Here “the first two models” means these two controller arms, not two different
policy algorithms. After this gate passes, the same pair can be run for CPO,
CUP, or any of the eight supported algorithms.

## 1. Locked protocol

The scenario key is `unannounced_meal_60g`. For every simulated day, one extra
`60 g` carbohydrate event is planned for minute `900` (`15:00`). The event is:

- injected into pending carbohydrates after the policy and Shield have selected
  the current action and before `env.step`;
- never added to `scenario_meals`, so it is absent from pre-event meal
  look-ahead features;
- allowed to become indirectly visible later through COB, CGM, and the changed
  physiology;
- paired by literal episode seed and planned event list between the two arms.

“Unannounced” therefore means no advance signal, not permanently invisible.
The old `meal_shift_minutes` examples are announced schedule perturbations and
must not be used as this experiment.

Primary safety metrics use true physiological BG. Native CGM and the exposed
CGM seen by the policy and Shield remain saved as diagnostics.

### 1.1 Which repository paths belong to this experiment

The formal path is:

| Path | Role |
| --- | --- |
| `interface_new.py` | plan and inject hidden carbohydrates without changing `scenario_meals` |
| `scripts/author_v1_stress_hook.py` | define `unannounced_meal_60g`, reset paired seeds, and call injection before each environment step |
| `patches/glucoalg_v1_stress_eval.patch` | place the injection call after policy/Shield action selection |
| `jobs/evaluate_author_v1_t1d_stress.slurm` | run the pinned evaluator and record `v1_use_meal_hyper_check=false` |
| `scripts/run_author_v1_cpo_unannounced_meal_local.sh` | run and validate the first local two-arm pair |
| `scripts/submit_author_v1_unannounced_meal_pair.sh` | submit the two full SURF arms |

`experiments/02_meal_shift_sanity.py`, `experiments/06_test_meal.py`, and the
meal-shift portion of `test_interface.py` modify announced schedules. They are
simulator sanity checks, not unexpected-meal Shield experiments.

## 2. Hyperglycaemia option and the correct control

The main unexpected-meal protocol keeps the author-v1 predictive Shield option
`use_meal_hyper_check=false`. The evaluator records:

```text
v1_use_meal_hyper_check=false
v1_hyper_check_threshold_mgdl=250
```

The threshold is recorded even when the check is disabled; the presence of
`250` alone does not prove that the option was active.

This switch controls whether predicted hyperglycaemia masks meal-recommendation
actions inside the Shield. It does not turn outcome logging on or off: the
evaluator still records hyperglycaemic events when the switch is `false`.

For this protocol, use these comparisons:

```text
Shield effect under unexpected meals
  = predictive(false) + unexpected meal
    minus no Shield + unexpected meal

Unexpected-meal effect without Shield
  = no Shield + unexpected meal
    minus no Shield + matched nominal

Unexpected-meal effect with Shield
  = predictive(false) + unexpected meal
    minus predictive(false) + matched nominal
```

Do not use `author_v1_evaluation/predictive_mealhyper250_penalty10` as the
predictive control for this experiment. That separate 2026-09-30 run explicitly
enabled the meal-hyper check and did not use this hidden-meal stress harness.

If the research question later becomes “what does the hyper check itself do?”,
add a predictive `true` arm under both nominal and unexpected-meal conditions.
Then compare predictive `true` with predictive `false` at the same meal
condition. Do not silently replace arm B above.

## 3. Shared prerequisites

Complete the SURF setup and pinned-source steps in Sections 4.1 through 4.6 of
the sensor-noise tutorial. The hidden-meal run uses the same January source
archives, epoch-2441 policies, transition set, and BA-NODE predictor.

From the repository root, this focused preflight should pass:

```bash
(
  set -euo pipefail

  test -f "${RTIDM_VENV:-$HOME/venvs/rtidm}/bin/activate"
  test -f author_v1_GlucoAlg_50d3134.tar.gz
  test -f author_v1_GlucoSim_6522d49.tar.gz
  test -f author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json
  test -f patches/glucosim_v1_python313.patch
  test -f patches/glucosim_v1_stress_diagnostics.patch
  test -f patches/glucoalg_v1_stress_eval.patch
  test -f interface_new.py
  test -f scripts/author_v1_stress_hook.py
  test -x scripts/submit_author_v1_unannounced_meal_pair.sh
  test -x scripts/run_author_v1_cpo_unannounced_meal_local.sh
  test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/config.yaml
  test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth
  test "$(find external/GlucoAlg/saved_files/env_transitions \
    -type f -name '*.npz' | wc -l)" -eq 180

  echo "unannounced-meal preflight passed"
)
```

## 4. Run the first matched pair locally

The smallest useful end-to-end check runs CPO, `child#002`, policy seed `0`,
and two episodes in each arm. In the pinned stack, the no-Shield seed-22
episode terminates before the first `15:00` event; adding seed 23 ensures that
both arms actually receive at least one hidden meal:

```bash
bash scripts/run_author_v1_cpo_unannounced_meal_local.sh \
  2 0 10 author_v1_unannounced_meal_local_pair_2ep
```

Arguments are:

```text
EPISODES  REPLICATE_ID  LOGIT_PENALTY  RESULT_COLLECTION
```

The helper runs both arms, validates their one-cell metadata, checks that their
planned meals and literal episode seeds match, requires at least one delivered
hidden meal in each arm, and verifies that the predictive metadata says
`v1_use_meal_hyper_check=false`. Success ends with:

```text
Calibration structure and diagnostics: PASS
Hidden-meal delivery in both arms: PASS
Pairing validation: PASS
Local unannounced-meal comparison: PASS
```

Use a new result-collection name for every fresh attempt. The helper refuses to
mix a new run into an existing directory.

The two result leaves are:

```text
author_v1_unannounced_meal_local_pair_2ep/
  unannounced_meal_60g/replicate0/no_shield/t1d/CPO/
  unannounced_meal_60g/replicate0/predictive_penalty10/t1d/CPO/
```

Inspect the predictive setting directly when needed:

```bash
rg '^v1_use_meal_hyper_check=' \
  author_v1_unannounced_meal_local_pair_2ep/unannounced_meal_60g/replicate0/\
predictive_penalty10/t1d/CPO/child#002/seed0/with_shield_10/run_metadata.txt
```

Also inspect the scenario record:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("author_v1_unannounced_meal_local_pair_2ep").glob(
    "unannounced_meal_60g/replicate0/*/t1d/CPO/child#002/seed0/*/"
    "stress_episode_metadata.json"
))
for path in paths:
    payload = json.loads(path.read_text())
    print(path)
    print("  scenario:", payload["stress_scenario"])
    for episode in payload["episodes"]:
        print("  seed:", episode["episode_seed"])
        print("    planned:", episode["planned_unannounced_meals"])
        print("    delivered:", episode["delivered_unannounced_meals"])
PY
```

A controller can terminate before all seven daily events. Delivered counts may
therefore differ, but the planned list and episode seed must match.

## 5. Submit a full CPO pair on SURF

After the one-cell gate passes, submit both CPO arms for all 81
patient/checkpoint cells:

```bash
bash scripts/submit_author_v1_unannounced_meal_pair.sh \
  CPO 1 0 10 author_v1_unannounced_meal_evaluation
```

Arguments are:

```text
ALGORITHM  EPISODES  REPLICATE_ID  LOGIT_PENALTY  RESULT_COLLECTION
```

The helper submits two smoke jobs at array index `0` and two dependent full
arrays at indices `1-80`. It prints and records all four Slurm job IDs. It also
prints `Predictive Shield meal-hyper check: false` before submission.

Monitor without resubmitting:

```bash
bash scripts/check_author_v1_unannounced_meal_status.sh
```

The checker takes one queue/accounting snapshot and exits. It never loops or
sleeps. When all 81 cells per arm are present, the same command also runs both
aggregations and the pairing validator. Optional arguments are:

```text
RESULT_COLLECTION  REPLICATE_ID  ALGORITHM  EPISODES  LOGIT_PENALTY
```

To run another algorithm, change only the algorithm while retaining the same
replicate id and protocol, for example:

```bash
bash scripts/submit_author_v1_unannounced_meal_pair.sh \
  CUP 1 0 10 author_v1_unannounced_meal_evaluation
```

Supported algorithms are:

```text
CPO  CUP  FOCOPS  OnCRPO  PCPO  PPOLag  RCPO  TRPOLag
```

## 6. Validate the full 81-cell pair

Set the run values:

```bash
ALGORITHM=CPO
EPISODES=1
REPLICATE_ID=0
PENALTY=10
RESULT_COLLECTION=author_v1_unannounced_meal_evaluation
ROOT="$RESULT_COLLECTION/unannounced_meal_60g/replicate$REPLICATE_ID"
BASE_ROOT="$ROOT/no_shield/t1d/$ALGORITHM"
SHIELD_ROOT="$ROOT/predictive_penalty${PENALTY}/t1d/$ALGORITHM"
mkdir -p validation_reports
```

Validate each arm and the pairing:

```bash
python scripts/aggregate_author_v1_t1d.py \
  "$BASE_ROOT" \
  --expected-episodes "$EPISODES" \
  --algorithm "$ALGORITHM" \
  --condition no_shield \
  --stress-scenario unannounced_meal_60g \
  --replicate-id "$REPLICATE_ID" \
  --expected-result-collection "$RESULT_COLLECTION" \
  | tee "validation_reports/${ALGORITHM}_hidden60_r${REPLICATE_ID}_no_shield.txt"

python scripts/aggregate_author_v1_t1d.py \
  "$SHIELD_ROOT" \
  --expected-episodes "$EPISODES" \
  --algorithm "$ALGORITHM" \
  --condition predictive \
  --logit-penalty "$PENALTY" \
  --stress-scenario unannounced_meal_60g \
  --replicate-id "$REPLICATE_ID" \
  --expected-result-collection "$RESULT_COLLECTION" \
  | tee "validation_reports/${ALGORITHM}_hidden60_r${REPLICATE_ID}_predictive.txt"

python scripts/validate_author_v1_stress_pairing.py \
  "$BASE_ROOT" \
  "$SHIELD_ROOT" \
  --expected-episodes "$EPISODES" \
  --replicate-id "$REPLICATE_ID" \
  --logit-penalty "$PENALTY" \
  | tee "validation_reports/${ALGORITHM}_hidden60_r${REPLICATE_ID}_pairing.txt"
```

Do not report the comparison until both aggregations print `Validation: PASS`
and the pairing validator prints `Pairing validation: PASS`.

## 7. What to report

Report TIR, Risk Index, CV, early termination, and the number of delivered
hidden meals together. Record the repository commit, scenario, replicate id,
episode count, penalty, and the exact hyper-check metadata.

The primary comparison is arm B minus arm A under the same unexpected-meal
plan. For a meal-effect claim, additionally run or reuse the matched nominal
pair from the same stress harness and the same replicate id. Do not compare
against an older nominal collection with different seed handling or a different
hyper-check setting.

## 8. Final checklist

- [ ] Scenario is exactly `unannounced_meal_60g`.
- [ ] Planned events are daily `15:00`, `60 g` events for seven days.
- [ ] Events are injected post-action and pre-step, not placed in
      `scenario_meals`.
- [ ] No-Shield and predictive arms have matching episode seeds and planned
      event lists.
- [ ] Predictive metadata says `v1_use_meal_hyper_check=false`.
- [ ] Both one-cell checks pass before full arrays are interpreted.
- [ ] Both 81-cell aggregations and the pairing validator pass.
- [ ] The matched nominal comparator uses the same stress harness, replicate,
      and hyper-check setting.
- [ ] The separate hyper-250 run is not mixed into the primary comparison.
