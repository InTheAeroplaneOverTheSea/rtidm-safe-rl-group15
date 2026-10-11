# Author-v1 unannounced-meal Shield tutorial

Last updated: 2026-10-11

The initial hidden-meal-only gate and the formal validation-bridge submissions
are documented in
[`AUTHOR_V1_UNANNOUNCED_MEAL_RUN_STATUS.md`](AUTHOR_V1_UNANNOUNCED_MEAL_RUN_STATUS.md).

This tutorial is only for the paper-aligned unexpected-meal experiment. Sensor
noise, bias, drift, and dropout are documented separately in
[`AUTHOR_V1_STRESS_TEST_TUTORIAL.md`](AUTHOR_V1_STRESS_TEST_TUTORIAL.md).

> **Comparison boundary:** the formal comparison uses ten independent
> evaluation seeds (`22..31`) per patient/checkpoint cell. Do not subtract an
> unexpected-meal result directly from the older validation result: the stress
> harness corrects episode-key reset semantics. The formal bridge therefore
> reruns both `nominal` and `unannounced_meal_60g` with those same ten literal
> seeds through the same stress harness. Compare old validation with bridge
> nominal only to quantify the harness change; estimate the meal effect only as
> hidden meal minus the matched bridge nominal.

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
| `scripts/check_author_v1_unannounced_meal_status.sh` | check one algorithm once and validate it when complete |
| `scripts/check_author_v1_unannounced_meal_person_ac_status.sh` | check the Person A/C algorithms with one shared Slurm snapshot |
| `scripts/submit_author_v1_unannounced_meal_validation_bridge.sh` | submit matched nominal and hidden-meal pairs for all eight algorithms |
| `scripts/check_author_v1_unannounced_meal_validation_bridge_status.sh` | check the full bridge once and validate complete pairs |

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
  test -f configs/author_v1_hf_revisions.json
  test -f patches/glucosim_v1_python313.patch
  test -f patches/glucosim_v1_stress_diagnostics.patch
  test -f patches/glucoalg_v1_stress_eval.patch
  test -f interface_new.py
  test -f scripts/author_v1_stress_hook.py
  test -x scripts/submit_author_v1_unannounced_meal_pair.sh
  test -x scripts/run_author_v1_cpo_unannounced_meal_local.sh
  test -x scripts/check_author_v1_unannounced_meal_status.sh
  test -x scripts/check_author_v1_unannounced_meal_person_ac_status.sh
  test -x scripts/submit_author_v1_unannounced_meal_validation_bridge.sh
  test -x scripts/check_author_v1_unannounced_meal_validation_bridge_status.sh
  test -f scripts/download_t1d_baseline_checkpoints.py
  test -f scripts/download_author_v1_transitions.py
  test -f jobs/download_t1d_baseline_checkpoints.slurm
  test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/config.yaml
  test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth
  test "$(find external/GlucoAlg/saved_files/env_transitions \
    -type f -name '*.npz' | wc -l)" -eq 180

  "${RTIDM_VENV:-$HOME/venvs/rtidm}/bin/python" \
    scripts/download_t1d_baseline_checkpoints.py \
    --glucoalg-dir external/GlucoAlg \
    --manifest author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json \
    --revisions-file configs/author_v1_hf_revisions.json \
    --algorithms CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag \
    --verify-only

  "${RTIDM_VENV:-$HOME/venvs/rtidm}/bin/python" \
    scripts/download_author_v1_transitions.py \
    --glucoalg-dir external/GlucoAlg \
    --revisions-file configs/author_v1_hf_revisions.json \
    --verify-only

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

## 5. Submit an 81-cell hidden-meal gate on SURF

After the one-cell gate passes, this command submits both CPO arms for all 81
patient/checkpoint cells. It is a complete hidden-meal matrix, but by itself it
is not a validation comparator because it has no matched nominal scenario:

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

An optional sixth argument selects `nominal` or `unannounced_meal_60g`; the
default is `unannounced_meal_60g`.

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

For the current stress-test allocation, Person A owns `CPO` and `CUP`, while
Person C owns `PCPO` and `PPOLag`. The CPO jobs already recorded in
[`AUTHOR_V1_UNANNOUNCED_MEAL_RUN_STATUS.md`](AUTHOR_V1_UNANNOUNCED_MEAL_RUN_STATUS.md)
must not be resubmitted. Submit the remaining three algorithms:

```bash
for algorithm in CUP PCPO PPOLag; do
  bash scripts/submit_author_v1_unannounced_meal_pair.sh \
    "$algorithm" 1 0 10 author_v1_unannounced_meal_evaluation
done
```

If the shared checkpoint verification in Section 4.6 is still waiting on one
Slurm download job, do not poll it and do not race the evaluations against it.
Attach both smoke jobs to that prerequisite:

```bash
STRESS_AFTEROK_JOB_ID="$checkpoint_job" \
  bash scripts/submit_author_v1_unannounced_meal_pair.sh \
  CUP 1 0 10 author_v1_unannounced_meal_evaluation
```

The dependent full arrays retain their existing dependency on the smoke jobs.
In a shared checkout, the checkpoint job must download all eight algorithms as
specified in Section 4.4; a partial download rewrites the shared manifest and
is only appropriate for a separate checkout.

Their combined checker takes exactly one queue snapshot and one accounting
snapshot, then performs file completeness checks and, for complete algorithms,
runs the validators. It never sleeps or polls:

```bash
bash scripts/check_author_v1_unannounced_meal_person_ac_status.sh
```

### 5.1 Submit the formal all-algorithm validation bridge

Use a fresh result collection containing both scenarios for all eight
algorithms. The helper deliberately enforces ten independent episode seeds per
cell. Set the two-hour Slurm time limit at submission without changing the
shared evaluation job:

```bash
SBATCH_TIMELIMIT=02:00:00 \
  bash scripts/submit_author_v1_unannounced_meal_validation_bridge.sh \
  10 0 10 author_v1_unannounced_meal_validation_bridge_10seed
```

This submits, for every algorithm:

```text
nominal: no Shield + predictive Shield
hidden 60 g: no Shield + predictive Shield
```

Each arm contains all 81 patient/checkpoint cells. The bridge contains
`8 algorithms x 2 scenarios x 2 arms x 81 cells x 10 seeds = 25920`
episode trajectories. Within each cell, the stress hook explicitly resets the
simulator with literal seeds `22` through `31`.
Do not reuse `author_v1_unannounced_meal_evaluation`; that collection is the
earlier hidden-meal-only gate.

Check the bridge once, without polling:

```bash
bash scripts/check_author_v1_unannounced_meal_validation_bridge_status.sh
```

## 6. Validate one 81-cell scenario pair

Set the run values:

```bash
ALGORITHM=CPO
EPISODES=10
REPLICATE_ID=0
PENALTY=10
RESULT_COLLECTION=author_v1_unannounced_meal_validation_bridge_10seed
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

The Shield comparison is arm B minus arm A under the same scenario. For a meal
effect, use only the two scenarios in the bridge collection:

```text
no-Shield meal effect = hidden no Shield - bridge nominal no Shield
predictive meal effect = hidden predictive - bridge nominal predictive
```

The older validation collection is not the meal-effect denominator because it
used different simulator-key reset semantics. Report bridge nominal minus old
validation separately as a harness-change diagnostic.

## 8. Final checklist

- [ ] The formal bridge contains both `nominal` and `unannounced_meal_60g`.
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
- [ ] Both scenarios contain ten detailed episode rows per cell with literal
      corrected simulator seeds `22..31`.
- [ ] The separate hyper-250 run is not mixed into the primary comparison.
