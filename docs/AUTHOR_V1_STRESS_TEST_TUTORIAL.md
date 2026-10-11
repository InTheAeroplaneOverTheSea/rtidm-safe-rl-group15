# Author-v1 Shield sensor-noise stress-test tutorial for SURF

Last updated: 2026-10-11

This is the step-by-step team tutorial for running the author-v1 T1D policies
with and without the predictive Shield under matched sensor-noise stress. It
follows the same operational style as the baseline validation tutorial: a
teammate should only need an assigned algorithm name, access to SURF, and the
commands below.

The workflow keeps the following artifacts pinned:

- January author-v1 GlucoAlg and GlucoSim source;
- January policy checkpoints at epoch `2441`;
- the January transition dataset;
- the group's BA-NODE predictor;
- the same patient/checkpoint matrix for no Shield and predictive Shield.

The unexpected-meal study is intentionally outside this matrix. Use
[`AUTHOR_V1_UNANNOUNCED_MEAL_TUTORIAL.md`](AUTHOR_V1_UNANNOUNCED_MEAL_TUTORIAL.md)
for its protocol, comparison groups, and commands.

## 1. Short version: run calibration first

Do not submit the full eight-algorithm matrix yet. After completing the shared
SURF setup in Section 4, the coordinator submits only the one-cell CPO
calibration:

```bash
cd "$HOME/projects/rtidm-safe-rl-group15"
git switch main
git pull --ff-only

bash scripts/submit_author_v1_cpo_stress_calibration.sh 10 0 10
```

This submits eight scenarios times two conditions, all restricted to array cell
`0` (`child#002`, CPO policy seed `0`). Each job runs ten literal episode seeds
`22` through `31`. It does not release cells `1-80` and does not submit any
other algorithm. The calibration helper requests eight hours per job by
default, overriding the evaluation job's one-hour full-matrix default. To use a
different limit explicitly, set `CALIBRATION_TIME_LIMIT` in
`[days-]HH:MM:SS` format.

Do not repeatedly monitor the jobs. Record the printed job IDs, log out, and
check them later using the commands in Section 8. After they finish, run:

```bash
python scripts/analyze_author_v1_cpo_stress_calibration.py \
  author_v1_stress_calibration
```

Only after the diagnostics pass and the drift severity is frozen should the
team use the full-matrix helpers.

## 2. What stress is actually applied

The intended primary matrix contains six canonical sensor scenarios, including
the matched nominal control:

| Scenario | Intervention | Status before calibration |
| --- | --- | --- |
| `nominal` | unchanged author-v1 noise configuration | fixed |
| `gaussian_sigma15` | replace minute-level Gaussian sigma `1.5` with `15 mg/dL` | fixed |
| `bias_positive` | set constant additive CGM bias to `+10 mg/dL` | candidate fixed value |
| `bias_negative` | set constant additive CGM bias to `-10 mg/dL` | candidate fixed value |
| `random_walk_drift` | set per-minute scale-walk sigma to `0.003` | provisional |
| `dropout_markov_v1` | five-minute Markov gaps, entry `0.003`, recovery `0.2917` | candidate fixed model |

`sensor_sigma15` remains an accepted input alias, but scripts normalize it to
`gaussian_sigma15`; it must not create a separate result set. Calibration also
uses `drift_sigma0025_calibration` and `drift_sigma004_calibration`. These two
keys are pilot-only and are not primary experiment conditions.

### 2.1 Native timing and nominal noise

The pinned simulator updates its sensor once per internal minute and averages
five minute-level CGM values into each controller observation. Nominal already
uses Gaussian sigma `1.5`, multiplicative random-walk sigma `0.002` per minute,
and independent native dropout probability `0.01` per minute. Native dropout
is forward-filled before the five-minute average; it does not expose a NaN to
the controller.

### 2.2 Gaussian and bias

`gaussian_sigma15` replaces the internal minute-level sigma with `15`; it does
not add another noise layer. Ignoring clipping, averaging reduces that white
component in the five-minute controller observation to approximately
`15/sqrt(5) = 6.7 mg/dL`. The name refers to the configured simulator field,
not the realized output SD.

The bias scenarios change only `cgm_bias_mgdl`. Because the simulator then
multiplies by its existing scale walk, configured `+10` or `-10` need not
produce an exact ten-unit difference at every output sample.

### 2.3 Drift calibration

The scale walk is updated `10080` times in a complete seven-day episode and is
clipped to `[0.9, 1.1]`. The calibration compares `0.0025`, `0.003`, and
`0.004` with nominal `0.002`. Freeze a value using scale-boundary occupancy,
sensor clipping, and realized error statistics before examining which setting
makes the Shield look better or worse.

### 2.4 Markov dropout

`dropout_markov_v1` is a separate five-minute observation-layer state machine:

```text
P(available -> unavailable) = 0.003
P(unavailable -> available) = 0.2917
```

During a gap, exposed CGM is forward-filled and exposed trend is zero. The same
trace is supplied to the policy, predictive Shield, and GlucoSim's CGM-based
meal/bolus gates. True physiological BG is retained only for outcome scoring.
The Markov schedule uses an independent deterministic RNG stream, so it does
not consume GlucoSim's physiology or action RNG.

## 3. Evaluation matrix and exact pairing

The full study contains eight algorithms:

```text
CPO  CUP  FOCOPS  OnCRPO  PCPO  PPOLag  RCPO  TRPOLag
```

For each scenario and condition there are `27 patients x 3 policy seeds = 81`
cells. Ten seven-day evaluation episodes and two Shield conditions across all
six scenarios would total `77,760` episodes, so full submission requires an
explicit post-calibration decision.

For replicate zero, episode indices `0..9` use the literal seeds:

```text
22, 23, 24, 25, 26, 27, 28, 29, 30, 31
```

The raw JAX key is reset immediately before every episode. Replicate `r` uses
`22 + 10000*r + episode`, so different replicate blocks cannot overlap.
Shield/no-Shield pairs share the exact seed, normal meal schedule, sensor RNG,
and Markov gap plan. Closed-loop trajectories may diverge after their actions
differ.

Primary safety metrics use the five-minute mean of true physiological
`BG_hist`. Every controller CSV separately saves true BG, native CGM, exposed
CGM, scale, native-dropout count, injected-gap state, and clipping flags.

## 4. Shared SURF setup: rerun this before stress testing

The remote environment must be set up again after pulling the merged stress
workflow, even if baseline validation ran successfully last week. The setup is
safe to rerun: it reuses `$HOME/venvs/rtidm`, refreshes required packages, and
verifies/downloads the pinned January transitions.

One person may perform this section for a shared checkout. With separate
checkouts, every person repeats it. Submit installation and downloads through
Slurm; do not perform compute work or package installation directly on the
login node.

### 4.1 Log in and update the checkout

```bash
ssh <SURF_USER>@snellius.surf.nl
```

For an existing checkout:

```bash
cd "$HOME/projects/rtidm-safe-rl-group15"
git status --short
git switch main
git pull --ff-only
git submodule update --init --recursive
git branch --show-current
git rev-parse HEAD
```

The branch must print `main`. Record the commit hash with the final results.
Do not delete existing raw results merely to make `git status` empty.

For a fresh checkout:

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --recurse-submodules \
  https://github.com/InTheAeroplaneOverTheSea/rtidm-safe-rl-group15.git
cd rtidm-safe-rl-group15
git switch main
git submodule update --init --recursive
```

All later commands assume the shell remains at the repository root.

### 4.2 Recreate or refresh the Python environment

Submit the idempotent setup job:

```bash
setup_job_raw=$(sbatch --parsable jobs/setup_surf.slurm)
setup_job=${setup_job_raw%%;*}
echo "setup job: $setup_job"
sacct -X -j "$setup_job" --format=JobID,JobName,State,Elapsed,ExitCode
```

Wait until it reports `COMPLETED` and `0:0`. Check it again later with:

```bash
sacct -X -j "$setup_job" --format=JobID,JobName,State,Elapsed,ExitCode
```

If it fails:

```bash
less "slurm-rtidm-setup-${setup_job}.out"
```

The setup job creates or refreshes `${RTIDM_VENV:-$HOME/venvs/rtidm}` and
downloads the pinned transition revision. Verify it after completion:

```bash
source "${RTIDM_VENV:-$HOME/venvs/rtidm}/bin/activate"
python scripts/download_author_v1_transitions.py \
  --glucoalg-dir external/GlucoAlg \
  --verify-only

test "$(find external/GlucoAlg/saved_files/env_transitions \
  -type f -name '*.npz' | wc -l)" -eq 180
echo "environment and transitions verified"
```

If a previous setup attempt reported `Multiple top-level packages discovered`,
pull `main` and submit `jobs/setup_surf.slurm` again. Do not manually edit the
pinned GlucoAlg submodule.

### 4.3 Fetch the old source archives

The evaluator does not run the authors' moving default branches. Fetch the
exact author-v1 source snapshots:

```bash
source_job_raw=$(sbatch --parsable jobs/fetch_author_v1_sources.slurm)
source_job=${source_job_raw%%;*}
echo "source job: $source_job"
sacct -X -j "$source_job" --format=JobID,JobName,State,Elapsed,ExitCode
```

After it completes, verify:

```bash
sha256sum \
  author_v1_GlucoAlg_50d3134.tar.gz \
  author_v1_GlucoSim_6522d49.tar.gz
```

Expected values:

```text
e2f775bd0a75fbbb657628708e98efe6d684dd5109edf7268c6e396cb5aa253f  author_v1_GlucoAlg_50d3134.tar.gz
3e8f7e48e8606f7136bd372898fd855d84fc959a70736cdf4b8db34c88063ba0  author_v1_GlucoSim_6522d49.tar.gz
```

They correspond to:

```text
GlucoAlg  50d3134fd6f0e01019295be0d6b99f110f794753
GlucoSim  6522d49ee1e6c1fd34f287519a5d41df735bbd38
```

Every stress job checks both archive hashes before simulation.

### 4.4 Download all January policy checkpoints

In a shared checkout, one person should download all eight algorithms in one
job. Do not launch multiple partial downloads concurrently because they write
the same manifest.

```bash
checkpoint_job_raw=$(sbatch --parsable jobs/download_t1d_baseline_checkpoints.slurm)
checkpoint_job=${checkpoint_job_raw%%;*}
echo "checkpoint job: $checkpoint_job"
sacct -X -j "$checkpoint_job" --format=JobID,JobName,State,Elapsed,ExitCode
```

The downloader uses immutable revisions from
`configs/author_v1_hf_revisions.json`, not the authors' updated Hugging Face
`main`. After completion:

```bash
test -f author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json

for algorithm in CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag; do
  for cohort in child adolescent adult; do
    for seed in 0 1 2; do
      root="external/GlucoAlg/saved_models/t1d/$cohort/$algorithm/seed$seed"
      test -f "$root/config.json"
      test -f "$root/torch_save/epoch-2441.pt"
    done
  done
done
echo "all January checkpoints present"
```

If each validator has a separate checkout, the job accepts exact algorithm
names, for example:

```bash
sbatch jobs/download_t1d_baseline_checkpoints.slurm CUP FOCOPS OnCRPO
```

### 4.5 Install the shared BA-NODE predictor

The predictive-Shield checkpoint is stored in this repository and must be
copied into GlucoAlg's ignored `saved_files` directory:

```bash
predictor_dir=external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42
mkdir -p "$predictor_dir"
cp checkpoints/ba_node_b5_p24_h24/42/config.yaml "$predictor_dir/"
cp checkpoints/ba_node_b5_p24_h24/42/best_model.pth "$predictor_dir/"

sha256sum "$predictor_dir/config.yaml" "$predictor_dir/best_model.pth"
```

Expected values:

```text
b89f4c5959b9a6803fe7a672ab554786f87024f6c0ac9ea5055e69f8e1b5aa6c  config.yaml
f6fbbb357b59c787ff6ef61500ac83bec139bacf043d0f66e66bc142adf97ed2  best_model.pth
```

### 4.6 Final shared preflight

Run this block before anyone submits stress jobs:

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
  test -x scripts/submit_author_v1_cpo_stress_calibration.sh
  test -x scripts/analyze_author_v1_cpo_stress_calibration.py
  test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/config.yaml
  test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth
  test "$(find external/GlucoAlg/saved_files/env_transitions \
    -type f -name '*.npz' | wc -l)" -eq 180

  printf '%s  %s\n%s  %s\n' \
    e2f775bd0a75fbbb657628708e98efe6d684dd5109edf7268c6e396cb5aa253f \
    author_v1_GlucoAlg_50d3134.tar.gz \
    3e8f7e48e8606f7136bd372898fd855d84fc959a70736cdf4b8db34c88063ba0 \
    author_v1_GlucoSim_6522d49.tar.gz \
    | sha256sum --check --strict

  echo "shared stress preflight passed"
)
```

No output from an individual `test` means it passed. Stop if the final success
line is not printed.

## 5. Divide work without duplicate submissions

Use the current four-person `2 + 2 + 2 + 2` allocation:

| Validator | Assigned algorithms | Count |
| --- | --- | ---: |
| Person A | `CPO`, `CUP` | 2 |
| Person B | `FOCOPS`, `OnCRPO` | 2 |
| Person C | `PCPO`, `PPOLag` | 2 |
| Person D | `RCPO`, `TRPOLag` | 2 |

One person owns an algorithm across all six canonical scenarios. Stable
output paths are intentional; two people
submitting the same algorithm/scenario/replicate can overwrite or mix results.

Each validator can record their assignment in the current shell:

```bash
MY_ALGORITHMS=(CPO CUP)
printf '%s\n' "${MY_ALGORITHMS[@]}"
```

This variable disappears after logout and does not change the repository.

## 6. CPO calibration gate before the full team run

Submit exactly one patient/policy cell, ten episodes, both conditions:

```bash
bash scripts/submit_author_v1_cpo_stress_calibration.sh 10 0 10
```

The helper requests `08:00:00` per calibration job. This is intentional: a
predictive-Shield smoke episode that terminates early is not a safe basis for
estimating the runtime of ten complete seven-day episodes. Override only when
needed, for example:

```bash
CALIBRATION_TIME_LIMIT=10:00:00 \
  bash scripts/submit_author_v1_cpo_stress_calibration.sh 10 0 10
```

If the pinned archives and checkpoints are already present but the refreshed
environment setup is still queued, the calibration may be submitted now with
an `afterok` dependency instead of polling the setup job:

```bash
CALIBRATION_AFTEROK_JOB_ID="$setup_job" \
  bash scripts/submit_author_v1_cpo_stress_calibration.sh 10 0 10
```

Every calibration job will then start only if that one setup job completes
successfully. Do not use this shortcut until the source archives and CPO
checkpoint preflight in Section 4.6 has passed.

This produces 16 independent jobs: eight calibration scenarios times two
conditions. It deliberately does not use `submit_author_v1_stress_pair.sh`,
because that helper would release the remaining 80 cells after smoke success.

The calibration cell is:

```text
algorithm: CPO
patient: child#002
policy seed: 0
array index: 0
episode seeds: 22 through 31
```

After every job reaches `COMPLETED`, run:

```bash
python scripts/analyze_author_v1_cpo_stress_calibration.py \
  author_v1_stress_calibration
```

The analyzer checks the literal seeds, true-BG outcome source, required trace
columns, finite diagnostics, and scenario metadata. It prints realized sensor
error, minute-level scale-boundary occupancy, clipping, and gap count/duration.
`PASS` confirms structural correctness; a human must still
choose the drift strength from the three candidates without first selecting on
Shield benefit.

## 7. Submit assigned algorithms

Only after calibration approval, submit one assigned algorithm for all six
canonical scenarios:

```bash
ALGORITHM=CUP
EPISODES=1
REPLICATE_ID=0
PENALTY=10

bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" nominal "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" gaussian_sigma15 "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" bias_positive "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" bias_negative "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" random_walk_drift "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" dropout_markov_v1 "$EPISODES" "$REPLICATE_ID" "$PENALTY"
```

Arguments are:

```text
ALGORITHM  SCENARIO  EPISODES  REPLICATE_ID  LOGIT_PENALTY  [RESULT_COLLECTION]
```

Use repository algorithm spelling exactly. Canonical scenarios are:

```text
nominal  gaussian_sigma15  bias_positive  bias_negative
random_walk_drift  dropout_markov_v1
```

The default result collection is `author_v1_stress_evaluation`. Do not change
it independently across teammates.

If one coordinator deliberately owns all eight algorithms, the matrix helper
can submit one entire scenario:

```bash
bash scripts/submit_author_v1_stress_matrix.sh nominal 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh gaussian_sigma15 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh bias_positive 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh bias_negative 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh random_walk_drift 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh dropout_markov_v1 1 0 10
```

Do not run the matrix helper once per teammate; each invocation already loops
over all eight algorithms.

For additional independent repeats, change only the replicate id and keep the
same id for both conditions and all six scenarios, for example:

```bash
bash scripts/submit_author_v1_stress_pair.sh CPO gaussian_sigma15 1 1 10
```

Reusing replicate `0` intentionally reproduces the same disturbance seeds.

## 8. Monitor jobs and check outputs

For the one-cell calibration, the submission record is:

```bash
RECORD=author_v1_stress_calibration/submitted_jobs.tsv
column -t -s $'\t' "$RECORD"
JOB_IDS=$(tail -n +2 "$RECORD" | cut -f9 | sort -u | paste -sd, -)
squeue -j "$JOB_IDS"
sacct -X -j "$JOB_IDS" --format=JobID,JobName,State,ExitCode,Elapsed
```

It is not necessary to keep this command running. Log out and execute the same
`sacct` command later. A finished calibration contains 16 top-level jobs and no
dependent full arrays.

Every submission is appended to:

```text
author_v1_stress_evaluation/<scenario>/replicate<id>/submitted_jobs.tsv
```

For one scenario:

```bash
SCENARIO=gaussian_sigma15
REPLICATE_ID=0
RECORD="author_v1_stress_evaluation/$SCENARIO/replicate$REPLICATE_ID/submitted_jobs.tsv"

column -t -s $'\t' "$RECORD"
JOB_IDS=$(tail -n +2 "$RECORD" | cut -f9 | sort -u | paste -sd, -)
squeue -j "$JOB_IDS"
sacct -X -j "$JOB_IDS" \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

It is normal for full arrays to remain in a dependency state while smoke jobs
run. If a smoke job fails, its dependent full array does not start.

The array mapping is:

```text
0-26   child
27-53  adolescent
54-80  adult

array_index = cohort_index * 27 + (patient_number - 2) * 3 + policy_seed
cohort_index: child=0, adolescent=1, adult=2
```

For example, `adolescent#005`, seed `2` is index `38`.

Inspect a failed cell with the exact log name shown by `sacct` or list likely
logs with:

```bash
ls -1t slurm-*stress* slurm-*gauss15* slurm-*gapmarkov* 2>/dev/null | head
```

For one algorithm and scenario, count results after both arrays complete:

```bash
ALGORITHM=CPO
SCENARIO=gaussian_sigma15
REPLICATE_ID=0
ROOT="author_v1_stress_evaluation/$SCENARIO/replicate$REPLICATE_ID"
BASE_ROOT="$ROOT/no_shield/t1d/$ALGORITHM"
SHIELD_ROOT="$ROOT/predictive_penalty10/t1d/$ALGORITHM"

for result_root in "$BASE_ROOT" "$SHIELD_ROOT"; do
  echo "$result_root"
  printf '  summaries:       '
  find "$result_root" -type f -name summary_results.csv | wc -l
  printf '  details:         '
  find "$result_root" -type f -name detailed_results.csv | wc -l
  printf '  run metadata:    '
  find "$result_root" -type f -name run_metadata.txt | wc -l
  printf '  stress metadata: '
  find "$result_root" -type f -name stress_episode_metadata.json | wc -l
done
```

Every count must be `81`. A Slurm `COMPLETED` state alone is not proof that all
result files are present.

## 9. Validate and aggregate one algorithm

The validators use only the Python standard library and may run on the login
node after simulations finish.

Set the values once:

```bash
ALGORITHM=CPO
SCENARIO=gaussian_sigma15
EPISODES=1
REPLICATE_ID=0
PENALTY=10
RESULT_COLLECTION=author_v1_stress_evaluation

ROOT="$RESULT_COLLECTION/$SCENARIO/replicate$REPLICATE_ID"
BASE_ROOT="$ROOT/no_shield/t1d/$ALGORITHM"
SHIELD_ROOT="$ROOT/predictive_penalty${PENALTY}/t1d/$ALGORITHM"
mkdir -p validation_reports
```

Validate and aggregate no Shield:

```bash
python scripts/aggregate_author_v1_t1d.py \
  "$BASE_ROOT" \
  --expected-episodes "$EPISODES" \
  --algorithm "$ALGORITHM" \
  --condition no_shield \
  --stress-scenario "$SCENARIO" \
  --replicate-id "$REPLICATE_ID" \
  --expected-result-collection "$RESULT_COLLECTION" \
  | tee "validation_reports/${ALGORITHM}_${SCENARIO}_r${REPLICATE_ID}_no_shield.txt"
```

Validate and aggregate predictive Shield:

```bash
python scripts/aggregate_author_v1_t1d.py \
  "$SHIELD_ROOT" \
  --expected-episodes "$EPISODES" \
  --algorithm "$ALGORITHM" \
  --condition predictive \
  --logit-penalty "$PENALTY" \
  --stress-scenario "$SCENARIO" \
  --replicate-id "$REPLICATE_ID" \
  --expected-result-collection "$RESULT_COLLECTION" \
  | tee "validation_reports/${ALGORITHM}_${SCENARIO}_r${REPLICATE_ID}_predictive.txt"
```

Verify that the two conditions received exactly matched disturbances:

```bash
python scripts/validate_author_v1_stress_pairing.py \
  "$BASE_ROOT" \
  "$SHIELD_ROOT" \
  --expected-episodes "$EPISODES" \
  --logit-penalty "$PENALTY" \
  | tee "validation_reports/${ALGORITHM}_${SCENARIO}_r${REPLICATE_ID}_pairing.txt"
```

Both aggregation commands must include:

```text
Matrix: 81/81 cells, 81 episode rows
Checkpoint identities: 9/9 cohort/seed pairs
Stress episode seed sequences: 1/1
Validation: PASS
```

The predictive validation must additionally print:

```text
Predictor identities: 1/1
```

The pairing command must print:

```text
Paired stress cells: 81/81
Pairing validation: PASS
```

For multiple episodes, replace `81 episode rows` with
`81 * EPISODES`. Do not report any condition until all three validators pass.

The stress validator additionally checks:

- the scenario and replicate id;
- corrected episode-key mode;
- consistent hook/interface/patch hashes;
- exact episode seed sequences;
- sensor sigma `15.0` for `gaussian_sigma15`;
- constant bias `+10/-10` for the two bias scenarios;
- the exact per-minute drift sigma for drift scenarios;
- Markov entry/recovery probabilities and the planned gap schedule.

Repeat Section 9 for every canonical scenario after calibration approval.

## 10. Interpret whether the Shield breaks

For every algorithm, retain the aggregated TIR, Risk Index, CV, and early
termination count. Make two paired comparisons:

```text
Shield effect under stress
  = predictive Shield under stress - no Shield under the same stress

Stress effect for one condition
  = stressed result - matched nominal result
```

Metric directions are:

| Metric | Better direction | Concerning Shield-minus-no-Shield result |
| --- | --- | --- |
| TIR | higher | negative |
| Risk Index | lower | positive |
| CV | lower | positive |
| early termination | fewer | more Shield terminations |

Do not define a Shield break solely as lower mean TIR. Flag cells or aggregate
conditions where the Shield increases early termination, Risk Index, or
variability as well. Report the fraction of paired cells in which Shield
worsens each clinical metric, not only the overall mean.

The main control is the `nominal` run made by this stress workflow with the
same replicate id. Do not compare against an old nominal result produced with
the stateful fixed-zero-key behavior.

## 11. Retry failed cells only

Find failed array indices:

```bash
sacct -X -j <JOB_ID> \
  --format=JobID,JobName,State,ExitCode,Elapsed \
  --noheader
```

Read the matching `slurm-..._<ARRAY_INDEX>.out` file before retrying.

For example, retry no-Shield indices `17`, `38`, and `75`:

```bash
sbatch --array=17,38,75 \
  jobs/evaluate_author_v1_t1d_stress.slurm \
  "$ALGORITHM" no_shield "$SCENARIO" \
  "$EPISODES" "$REPLICATE_ID" "$PENALTY" "$RESULT_COLLECTION"
```

Retry the matching predictive-Shield cells:

```bash
sbatch --array=17,38,75 \
  jobs/evaluate_author_v1_t1d_stress.slurm \
  "$ALGORITHM" predictive "$SCENARIO" \
  "$EPISODES" "$REPLICATE_ID" "$PENALTY" "$RESULT_COLLECTION"
```

After retrying, rerun file counts and all three validation commands.

### Common failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| setup or imports fail | remote environment was not refreshed | pull `main` and rerun `sbatch jobs/setup_surf.slurm` |
| archive hash failure | wrong or partial author-v1 source archive | rerun `jobs/fetch_author_v1_sources.slurm` |
| checkpoint revision/hash failure | current Hugging Face assets replaced January files | rerun the pinned checkpoint download job; never bypass `--verify-only` |
| missing checkpoint manifest | checkpoint setup was not completed in this checkout | run `jobs/download_t1d_baseline_checkpoints.slurm` |
| predictive fails but no Shield passes | predictor or transitions missing | repeat Sections 4.2 and 4.5 |
| full array remains pending | its smoke dependency has not passed | inspect smoke state and log |
| exactly 80 outputs | smoke cell `0` is missing | inspect the smoke job and stable output path |
| pairing validator fails | mixed replicate/scenario/code or stale cell outputs | inspect metadata, remove only the incorrect cell output, and rerun that cell |
| results exist but validator reports wrong scenario | output directory reused with different arguments | choose a new result collection or repair the mismatched cells |

## 12. Team handoff template

For every algorithm and scenario, report:

```text
Algorithm:
Validator:
Date:
Repository commit:
Scenario:
Replicate id:
Episodes per cell:
Logit penalty:

Slurm jobs:
  no-Shield smoke:
  no-Shield full:
  predictive smoke:
  predictive full:

Completeness:
  no Shield: 81 summaries / 81 details / 81 run metadata / 81 stress metadata
  predictive: 81 summaries / 81 details / 81 run metadata / 81 stress metadata

Validation:
  no-Shield aggregation: PASS / FAIL
  predictive aggregation: PASS / FAIL
  paired disturbances: PASS / FAIL

Results (mean +/- sample SD):
  no Shield: TIR ..., Risk ..., CV ...
  predictive: TIR ..., Risk ..., CV ...
  Shield-minus-no-Shield: TIR ..., Risk ..., CV ...
  early termination: no Shield ... / predictive ...

Matched nominal comparison:
  no-Shield stress-minus-nominal: ...
  predictive stress-minus-nominal: ...

Potential Shield break indicators:
Notes / unresolved discrepancies:
```

Attach the three files written under `validation_reports/`.

## 13. Final checklist

- [ ] The remote checkout is on current `main` and the commit is recorded.
- [ ] `jobs/setup_surf.slurm` was rerun after pulling the stress workflow.
- [ ] Transition verification reports the pinned January revision and 180 files.
- [ ] Both author-v1 archive hashes match Section 4.3.
- [ ] Checkpoints come from immutable January revisions at epoch `2441`.
- [ ] The checkpoint manifest exists and verification passes.
- [ ] The BA-NODE predictor hashes match Section 4.5.
- [ ] The exact repository algorithm name is used.
- [ ] All six canonical sensor scenarios use the same replicate id.
- [ ] Both smoke jobs pass before the full team submission.
- [ ] Both conditions contain 81 cells and all four required output files per cell.
- [ ] Both aggregation validators print `Validation: PASS`.
- [ ] The paired validator prints `Pairing validation: PASS`.
- [ ] TIR, Risk Index, CV, and early termination are reported together.
- [ ] Stress results are compared with the new matched nominal, not an old baseline collection.
- [ ] Unexpected-meal runs follow the separate unexpected-meal tutorial.
- [ ] Experimental interface-only noise fields are not mixed into the formal scenarios.
- [ ] Raw outputs, archives, downloaded checkpoints, and Slurm logs are not committed to Git.

## 14. Generated files that stay outside Git

Do not commit:

```text
author_v1_GlucoAlg_50d3134.tar.gz
author_v1_GlucoSim_6522d49.tar.gz
author_v1_all_baselines_evaluation/
author_v1_stress_evaluation/
external/GlucoAlg/saved_models/
external/GlucoAlg/saved_files/
slurm-*.out
surf_environment_freeze.txt
```

Commit only agreed scripts, patches, documentation, and compact text reports.
Always inspect `git status --short` and stage exact paths; do not use
`git add .` in a checkout containing evaluation outputs.
