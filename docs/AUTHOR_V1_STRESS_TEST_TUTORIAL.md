# Author-v1 Shield stress-test tutorial for SURF

Last updated: 2026-10-11

This is the step-by-step team tutorial for running the author-v1 T1D policies
with and without the predictive Shield under matched sensor-noise and
unannounced-meal stress. It follows the same operational style as the baseline
validation tutorial: a teammate should only need an assigned algorithm name,
access to SURF, and the commands below.

The workflow keeps the following artifacts pinned:

- January author-v1 GlucoAlg and GlucoSim source;
- January policy checkpoints at epoch `2441`;
- the January transition dataset;
- the group's BA-NODE predictor;
- the same patient/checkpoint matrix for no Shield and predictive Shield.

Do not run the old announced meal-shift example as the meal stress in this
experiment. The formal meal scenario below is hidden from the policy and
Shield until after their current action has been selected.

## 1. Short version for teammates

One person completes the shared remote setup in Section 4. After that, a
validator assigned `CUP` can submit all three matched scenarios with:

```bash
cd "$HOME/projects/rtidm-safe-rl-group15"
git switch main
git pull --ff-only

ALGORITHM=CUP
bash scripts/submit_author_v1_stress_pair.sh "$ALGORITHM" nominal 1 0 10
bash scripts/submit_author_v1_stress_pair.sh "$ALGORITHM" sensor_sigma15 1 0 10
bash scripts/submit_author_v1_stress_pair.sh "$ALGORITHM" unannounced_meal_60g 1 0 10
```

Each command submits a no-Shield smoke job, a predictive-Shield smoke job, and
two dependent full arrays. Do not submit the same algorithm and scenario twice
into the same result collection.

After all cells finish, set `SCENARIO` and `ALGORITHM` and copy the three
validation commands from Section 9. The result is reportable only when both
condition validators and the paired-disturbance validator print `PASS`.

The short version assumes all checks in Section 4 already pass. A missing
archive, manifest, checkpoint, transition file, or predictor will deliberately
stop the evaluation job.

## 2. What stress is actually applied

The submission workflow exposes exactly three formal scenarios:

| Scenario | Intervention | Purpose |
| --- | --- | --- |
| `nominal` | no extra stress | matched control under the corrected seed handling |
| `sensor_sigma15` | set CGM observation-noise sigma to `15 mg/dL` | strong measurement noise |
| `unannounced_meal_60g` | add `60 g` at `15:00` every simulated day | surprise carbohydrate disturbance |

### 2.1 Nominal matched control

`nominal` does not add sensor or meal stress. It is still important to rerun
it with this workflow because the stress adapter resets the old simulator's
JAX key explicitly before every episode. Do not use an older baseline
collection as the numerical control for these stress runs.

### 2.2 Sensor noise

`sensor_sigma15` sets the simulator field `cgm_obs_sigma_mgdl` to exactly
`15.0`. It does not add 15 to the nominal sigma. At every observation the CGM
sensor receives independent zero-mean Gaussian measurement noise with standard
deviation `15 mg/dL`:

```text
observed CGM = underlying CGM + Normal(0, 15^2) mg/dL
```

This changes what the policy and predictive Shield observe. It does not
directly change the patient's physiological state.

This is an amplification of a noise mechanism that already exists in the
January simulator, not a new noise family. The nominal `NoiseConfig` already
contains:

```text
additive Gaussian CGM sigma       1.5 mg/dL
multiplicative scale random walk  step sigma 0.002, clipped to 0.9-1.1
CGM dropout probability           0.01 per step
```

The formal `sensor_sigma15` scenario changes only the first value from `1.5`
to `15.0`; the nominal random-walk drift and dropout remain active.

This test should not be described as disproving a claim that the Shield is
"Gaussian-noise-proof." The paper does not give an unconditional guarantee
against Gaussian observation noise. Its probabilistic theorem is conditional
on the learned predictor remaining one-sided `(epsilon, alpha)`-reliable over
the finite prediction horizon and applies only to candidate actions screened
by the Shield. Strong sensor noise can invalidate that predictor-error
assumption. The paper also notes that the implemented Shield can be sensitive
to small noise near its 60-80 mg/dL transition zone and disables predictive
verification there.

Therefore, treat `sensor_sigma15` as a ten-times-nominal severity and
degradation test. It remains useful empirically, but it is not the most novel
candidate for exposing an unmodeled sensor failure. Persistent calibration
bias, scale drift, and dropout should be registered and validated as separate
future scenarios before the team uses them. The unannounced meal is the
current workflow's most direct test of a failure mode explicitly identified by
the paper.

### 2.3 Unannounced meal

`unannounced_meal_60g` preserves the normally sampled meal schedule and adds
one extra `60 g` meal at `15:00` on each simulated day. The event order is:

```text
current observation
  -> policy proposes action
  -> Shield accepts or changes action
  -> evaluator injects 60 g into the pending carbohydrate state
  -> env.step executes
  -> later COB and CGM reveal the physiological consequence
```

The hidden meal is never inserted into `scenario_meals`. Before the event, it
cannot appear in:

- `time_until_meal`;
- `next_meal_size`;
- the pre-bolus flag;
- any earlier observation used by the policy or Shield.

A complete author-v1 episode is seven simulated days (`2016` five-minute
steps), so a complete episode plans seven hidden meals. If an episode
terminates early, fewer than seven may actually be delivered; that is a valid
outcome and is recorded in `stress_episode_metadata.json`.

### 2.4 Other stress fields Sherlock implemented

The simulator interface supports more than the single Gaussian-noise scenario.
The following fields exist in `StressConfig`, but they are not currently
registered as formal command-line scenarios in the author-v1 stress workflow:

| Interface field | Meaning | Formal workflow status |
| --- | --- | --- |
| `cgm_bias_mgdl` | constant CGM calibration offset | implemented in the interface; not exposed by the submit scripts |
| `cgm_rw_sigma_bias` | random-walk increment of the multiplicative CGM scale | implemented in the interface; not exposed by the submit scripts |
| `cgm_dropout_prob` | per-step CGM measurement dropout probability | implemented in the interface; not exposed by the submit scripts |
| `cgm_use_ou`, `cgm_ou_sigma` | OU process noise in physiological glucose-state exchange | experimental; despite the field prefix, this is not merely CGM sensor noise |
| `kabs_scale` | carbohydrate-absorption dynamics shift | implemented physiological OOD; not a sensor-noise scenario |
| announced meal shift/jitter/scale | changes the scheduled meals visible in look-ahead fields | simulator sanity tests only, not the paper-aligned surprise meal |

Do not add one of these fields ad hoc to a team run. A new formal scenario
should first receive a fixed name and magnitude, be added to the stress
registry, metadata validator, documentation, and no-Shield/Shield pairing test.
Otherwise different teammates can silently run different experiments under
the same label.

Relevant paper sections are the
[conditional probabilistic bound](https://arxiv.org/html/2601.21094v1#S5.SS4),
the [implemented gating mechanism](https://arxiv.org/html/2601.21094v1#S5.SS2),
and the [limitations and future-work discussion](https://arxiv.org/html/2601.21094v1#S7).

## 3. Evaluation matrix and exact pairing

The eight repository algorithm names are:

```text
CPO  CUP  FOCOPS  OnCRPO  PCPO  PPOLag  RCPO  TRPOLag
```

The paper calls `OnCRPO` simply `CRPO`; always use `OnCRPO` in commands and
paths.

For one algorithm, one scenario, and one condition, the matrix is:

```text
3 cohorts (child, adolescent, adult)
x 9 unseen patients per cohort (#002 through #010)
x 3 policy-training seeds (0, 1, 2)
x 1 evaluation episode initially
= 81 cells
```

Every scenario has two conditions:

```text
81 no-Shield cells + 81 predictive-Shield cells = 162 cells
```

One complete eight-algorithm scenario therefore submits `1,296` array tasks.
The three-scenario matched experiment submits `3,888` tasks. Run one CPO pair
first instead of launching the entire matrix immediately.

Before every episode, the evaluator creates a fresh condition-independent
seed:

```text
SeedSequence([base_seed=22, replicate_id, episode, 20261011])
```

The algorithm and Shield condition are intentionally absent. Matched
no-Shield and Shield cells therefore start with the same:

- episode seed;
- normal meal schedule;
- sensor-noise stream;
- planned hidden meals.

This fixes the January simulator's stateful fixed-zero-key issue. Later
episodes remain paired even if an earlier Shield and no-Shield episode ended at
different steps. Multi-episode runs are supported, although the first team
run should stay at one episode per cell for faster diagnosis.

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
  test -f patches/glucoalg_v1_stress_eval.patch
  test -f interface_new.py
  test -f scripts/author_v1_stress_hook.py
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

One person owns an algorithm across `nominal`, `sensor_sigma15`, and
`unannounced_meal_60g`. Stable output paths are intentional; two people
submitting the same algorithm/scenario/replicate can overwrite or mix results.

Each validator can record their assignment in the current shell:

```bash
MY_ALGORITHMS=(CPO CUP)
printf '%s\n' "${MY_ALGORITHMS[@]}"
```

This variable disappears after logout and does not change the repository.

## 6. Pilot CPO before the full team run

The coordinator should first run one episode for CPO, replicate zero, penalty
10:

```bash
bash scripts/submit_author_v1_stress_pair.sh CPO nominal 1 0 10
bash scripts/submit_author_v1_stress_pair.sh CPO sensor_sigma15 1 0 10
bash scripts/submit_author_v1_stress_pair.sh CPO unannounced_meal_60g 1 0 10
```

Each command submits four jobs:

1. no-Shield smoke cell `0`;
2. predictive-Shield smoke cell `0`;
3. no-Shield cells `1-80`, released only after its smoke succeeds;
4. predictive-Shield cells `1-80`, released only after its smoke succeeds.

The smoke cell is always:

```text
child#002, policy seed 0, array index 0
```

Do not start all eight algorithms until at least one no-Shield and Shield smoke
pair completes with exit code `0:0` and produces all four expected output
files.

## 7. Submit assigned algorithms

For one assigned algorithm:

```bash
ALGORITHM=CUP
EPISODES=1
REPLICATE_ID=0
PENALTY=10

bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" nominal "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" sensor_sigma15 "$EPISODES" "$REPLICATE_ID" "$PENALTY"
bash scripts/submit_author_v1_stress_pair.sh \
  "$ALGORITHM" unannounced_meal_60g "$EPISODES" "$REPLICATE_ID" "$PENALTY"
```

Arguments are:

```text
ALGORITHM  SCENARIO  EPISODES  REPLICATE_ID  LOGIT_PENALTY  [RESULT_COLLECTION]
```

Use repository algorithm spelling exactly. Supported scenarios are only:

```text
nominal  sensor_sigma15  unannounced_meal_60g
```

The default result collection is `author_v1_stress_evaluation`. Do not change
it independently across teammates.

If one coordinator deliberately owns all eight algorithms, the matrix helper
can submit one entire scenario:

```bash
bash scripts/submit_author_v1_stress_matrix.sh nominal 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh sensor_sigma15 1 0 10
bash scripts/submit_author_v1_stress_matrix.sh unannounced_meal_60g 1 0 10
```

Do not run the matrix helper once per teammate; each invocation already loops
over all eight algorithms.

For additional independent repeats, change only the replicate id and keep the
same id for both conditions and all three scenarios, for example:

```bash
bash scripts/submit_author_v1_stress_pair.sh CPO sensor_sigma15 1 1 10
```

Reusing replicate `0` intentionally reproduces the same disturbance seeds.

## 8. Monitor jobs and check outputs

Every submission is appended to:

```text
author_v1_stress_evaluation/<scenario>/replicate<id>/submitted_jobs.tsv
```

For one scenario:

```bash
SCENARIO=sensor_sigma15
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
ls -1t slurm-*stress* slurm-*sensor15* slurm-*hidden60* 2>/dev/null | head
```

For one algorithm and scenario, count results after both arrays complete:

```bash
ALGORITHM=CPO
SCENARIO=sensor_sigma15
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
SCENARIO=sensor_sigma15
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
- sensor sigma `15.0` for `sensor_sigma15`;
- seven planned daily `15:00`, `60 g` events for the hidden-meal scenario.

It is valid for no Shield and Shield to deliver different numbers of hidden
meals if one controller terminates earlier. Their planned disturbance, seed,
and normal meal schedule must still match.

Repeat Section 9 for `nominal`, `sensor_sigma15`, and
`unannounced_meal_60g`.

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
- [ ] `nominal`, `sensor_sigma15`, and `unannounced_meal_60g` use the same replicate id.
- [ ] Both smoke jobs pass before the full team submission.
- [ ] Both conditions contain 81 cells and all four required output files per cell.
- [ ] Both aggregation validators print `Validation: PASS`.
- [ ] The paired validator prints `Pairing validation: PASS`.
- [ ] TIR, Risk Index, CV, and early termination are reported together.
- [ ] Stress results are compared with the new matched nominal, not an old baseline collection.
- [ ] Announced meal shifts are not described as unannounced meals.
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
