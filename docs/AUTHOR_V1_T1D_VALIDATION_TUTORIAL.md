# Author-v1 T1D validation tutorial

Last updated: 2026-09-30

This is a practical tutorial for the three teammates who will validate the
eight T1D Safe-RL algorithms in a `3 + 3 + 2` split. It is not a project
handoff or a summary of one person's work. Each validator should be able to
start with their assigned algorithm names, follow the same procedure, and
produce comparable evidence.

The workflow uses Snellius, Slurm, the public policy checkpoints, the January
author-v1 code stack, and the group's trained BA-NODE predictor. It validates
both the unshielded policy and the predictive-Shield condition.

## 1. What each person is validating

The repository names for the eight algorithms are:

```text
CPO  CUP  FOCOPS  OnCRPO  PCPO  PPOLag  RCPO  TRPOLag
```

The paper calls `OnCRPO` simply `CRPO`. Use `OnCRPO` in every command and file
path; use `CRPO` only when referring to the paper's table label.

Fill in the agreed team allocation before submitting anything:

| Validator | Assigned algorithms | Expected count |
|---|---|---:|
| Person A | `CPO`, `CUP`, `FOCOPS` | 3 |
| Person B | `OnCRPO`, `PCPO`, `PPOLag` | 3 |
| Person C | `RCPO`, `TRPOLag` | 2 |

One person owns an algorithm end to end. Do not have two people submit the same
algorithm into the same result directory: the jobs intentionally use stable
paths, so duplicate runs can overwrite or mix outputs.

For one algorithm and one condition, the validation matrix is:

```text
3 policy-training seeds
× 3 cohorts (child, adolescent, adult)
× 9 unseen patients per cohort (#002 through #010)
× 1 evaluation episode initially
= 81 result cells
```

Each algorithm is evaluated under two conditions:

```text
81 no-Shield cells + 81 predictive-Shield cells = 162 cells per algorithm
```

The full `3 + 3 + 2` assignment therefore produces 1,296 one-episode result
cells. A later ten-episode confirmation is a separate experiment; do not mix
one-episode and ten-episode files in the same validation report.

In this evaluator, a complete episode is `2016` five-minute steps, or seven
simulated days. Some unsafe trajectories terminate earlier; the validator
reports those separately rather than discarding them.

Keep the following configuration fixed across all eight algorithms:

| Item | Required value |
|---|---|
| patient type | `t1d` |
| policy training patient | `#001` in each cohort |
| evaluation patients | `#002-#010` in each cohort |
| policy-training seeds | `0`, `1`, `2` |
| checkpoint | `epoch-2441.pt` |
| evaluation seed field | `22` |
| initial episode count | `1` |
| evaluator device | author-v1 CPU mode |
| Shield type | predictive |
| logit penalty | `10` |
| author-v1 meal-hyper check | `false` |
| author-v1 hyper threshold field | `250 mg/dL` |
| predictor | group BA-NODE, `ba_node_b5_p24_h24/42` |

The `250 mg/dL` value is still recorded when the meal-hyper check is false; it
does not mean that the filter is active.

## 2. What “validation” means here

Validation has four layers. Passing only the first layer is not enough.

1. **Provenance:** every cell used the intended source commits, checkpoint
   epoch, policy seed, predictor, Shield settings, and episode count.
2. **Completeness:** all 81 cells exist for each condition, and each contains a
   summary, episode-level detail, and metadata.
3. **Aggregation:** every teammate aggregates in the same order:
   `episodes -> patient -> cohort -> policy-training seed -> final mean/SD`.
4. **Numerical comparison:** the reproduced no-Shield, Shield, and paired
   changes are compared with the same paper revision and table. Differences
   are recorded, not silently explained away.

A useful status vocabulary is:

- **structural PASS:** all 81 cells and provenance checks pass;
- **numerically close:** paper comparison is close enough to support continued
  investigation, with the exact delta reported;
- **directionally consistent:** the sign of the Shield effect agrees, but the
  magnitude does not;
- **mismatch / unresolved:** the baseline, effect direction, or provenance is
  inconsistent and must be investigated.

Do not turn “close” into an arbitrary pass/fail tolerance after seeing the
result. Always report the actual differences.

## 3. Shared setup: do this once

If the team uses one shared checkout, one person should perform this section
and tell the other validators when it is complete. If each person uses a
separate checkout, each person must repeat it.

### 3.1 Clone the repository

Connect to Snellius using the SURF account assigned to you:

```bash
ssh <SURF_USER>@snellius.surf.nl
```

Then clone and enter the project:

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --recurse-submodules \
  https://github.com/InTheAeroplaneOverTheSea/rtidm-safe-rl-group15.git
cd rtidm-safe-rl-group15
git switch Yule-shield-test
git submodule update --init --recursive
```

Before every validation session, confirm that you are in the correct checkout:

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short
```

Run `sbatch`, `squeue`, `sacct`, small file checks, and aggregation on the login
node. Do not run simulations or package installation directly on the login
node.

### 3.2 Create the environment and fetch transitions

Submit the setup job:

```bash
setup_job_raw=$(sbatch --parsable jobs/setup_surf.slurm)
setup_job=${setup_job_raw%%;*}
echo "$setup_job"
sacct -X -j "$setup_job" --format=JobID,State,Elapsed,ExitCode
```

Continue only after the job is `COMPLETED` with exit code `0:0`. If it fails,
read its Slurm log before resubmitting:

```bash
less "slurm-rtidm-setup-${setup_job}.out"
```

The setup job creates `$HOME/venvs/rtidm`, installs the dependencies, and
downloads the transition data. Verify the expected count:

```bash
test "$(find external/GlucoAlg/saved_files/env_transitions \
  -type f -name '*.npz' | wc -l)" -eq 180
```

### 3.3 Fetch the exact author-v1 source archives

```bash
source_job_raw=$(sbatch --parsable jobs/fetch_author_v1_sources.slurm)
source_job=${source_job_raw%%;*}
echo "$source_job"
sacct -X -j "$source_job" --format=JobID,State,Elapsed,ExitCode
```

After completion, verify the archives:

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

These archives pin:

```text
GlucoAlg  50d3134fd6f0e01019295be0d6b99f110f794753
GlucoSim  6522d49ee1e6c1fd34f287519a5d41df735bbd38
```

### 3.4 Download all eight algorithms' public checkpoints

In a shared checkout, run this once for all eight algorithms so that the
manifest is complete and concurrent downloads cannot race:

```bash
checkpoint_job_raw=$(sbatch --parsable jobs/download_t1d_baseline_checkpoints.slurm)
checkpoint_job=${checkpoint_job_raw%%;*}
echo "$checkpoint_job"
sacct -X -j "$checkpoint_job" --format=JobID,State,Elapsed,ExitCode
```

The job installs epoch `2441` for three cohorts and three policy seeds. Verify
the complete set:

```bash
for algorithm in CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag; do
  for cohort in child adolescent adult; do
    for seed in 0 1 2; do
      root="external/GlucoAlg/saved_models/t1d/$cohort/$algorithm/seed$seed"
      test -f "$root/config.json"
      test -f "$root/torch_save/epoch-2441.pt"
    done
  done
done
echo "all checkpoints present"
```

Checkpoint revisions and installed hashes are recorded in:

```text
author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json
```

If validators use separate checkouts, a person may download only their own
algorithms by adding exact algorithm names after the Slurm script. For example:

```bash
sbatch jobs/download_t1d_baseline_checkpoints.slurm CUP FOCOPS OnCRPO
```

Do not do partial downloads concurrently in one checkout because each job
writes the same manifest path.

### 3.5 Install and verify the shared BA-NODE predictor

The authors' final predictive-Shield checkpoint is not public. This workflow
uses the group-trained model. Obtain the two files from the team's shared
storage and place them at:

```text
external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/
  config.yaml
  best_model.pth
```

Verify them:

```bash
sha256sum \
  external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/config.yaml \
  external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth
```

Expected values:

```text
b89f4c5959b9a6803fe7a672ab554786f87024f6c0ac9ea5055e69f8e1b5aa6c  config.yaml
f6fbbb357b59c787ff6ef61500ac83bec139bacf043d0f66e66bc142adf97ed2  best_model.pth
```

Stop here if an archive, transition file, checkpoint, or predictor hash is
wrong. Running the full arrays with mismatched inputs only creates unusable
results.

## 4. Per-person workflow

Repeat this section once for each assigned algorithm. The examples use `CUP`;
replace it with one exact repository name from Section 1.

### 4.1 Record your assignment locally in the shell

For a three-algorithm assignment:

```bash
MY_ALGORITHMS=(CUP FOCOPS OnCRPO)
printf '%s\n' "${MY_ALGORITHMS[@]}"
```

For a two-algorithm assignment, list only two names. This shell variable is
only a convenience and disappears when you log out.

### 4.2 Preflight one algorithm

```bash
ALGORITHM=CUP

for cohort in child adolescent adult; do
  for seed in 0 1 2; do
    root="external/GlucoAlg/saved_models/t1d/$cohort/$ALGORITHM/seed$seed"
    test -f "$root/config.json"
    test -f "$root/torch_save/epoch-2441.pt"
  done
done

test -f author_v1_GlucoAlg_50d3134.tar.gz
test -f author_v1_GlucoSim_6522d49.tar.gz
test -f patches/glucosim_v1_python313.patch
test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/config.yaml
test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth
test "$(find external/GlucoAlg/saved_files/env_transitions \
  -type f -name '*.npz' | wc -l)" -eq 180

echo "$ALGORITHM preflight passed"
```

No output from a `test` command means it passed. If the shell stops before the
final line, inspect the last path instead of submitting jobs.

### 4.3 Submit the no-Shield and predictive-Shield pair

```bash
./scripts/submit_author_v1_baseline_pair.sh "$ALGORITHM"
```

The helper submits four jobs:

1. no-Shield smoke test: array cell `0`;
2. predictive-Shield smoke test: array cell `0`;
3. no-Shield full array: cells `1-80`, released only if its smoke passes;
4. predictive-Shield full array: cells `1-80`, released only if its smoke
   passes.

The helper prints all four job IDs and appends them to:

```text
author_v1_all_baselines_evaluation/submitted_jobs.tsv
```

Do not immediately submit every assigned algorithm before checking at least one
smoke pair. A smoke failure usually indicates a shared setup problem that would
affect every algorithm.

### 4.4 Check the smoke tests

Use the job IDs printed by the helper:

```bash
sacct -X -j <BASE_SMOKE_ID>,<SHIELD_SMOKE_ID> \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

Both smoke jobs must be `COMPLETED 0:0`. Inspect their logs:

```bash
less slurm-v1-cup-base-smoke-<BASE_SMOKE_ID>_0.out
less slurm-v1-cup-shield-smoke-<SHIELD_SMOKE_ID>_0.out
```

Job names use a lowercase algorithm slug. For example, `OnCRPO` becomes
`oncrpo` and `PPOLag` becomes `ppolag`.

The smoke cell is always:

```text
cohort=child, patient=child#002, policy_seed=0, array_index=0
```

Confirm that it produced both conditions:

```bash
test -f \
  "author_v1_all_baselines_evaluation/no_shield/t1d/$ALGORITHM/child#002/seed0/no_shield/summary_results.csv"

test -f \
  "author_v1_all_baselines_evaluation/predictive_penalty10/t1d/$ALGORITHM/child#002/seed0/with_shield_10/summary_results.csv"
```

### 4.5 Monitor the full arrays

To view every recorded team job and result count:

```bash
./scripts/check_author_v1_baseline_jobs.sh "$PWD"
```

For only your current pair:

```bash
squeue -j <BASE_FULL_ID>,<SHIELD_FULL_ID>
sacct -X -j <BASE_FULL_ID>,<SHIELD_FULL_ID> \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

Slurm may keep the full array in a dependency state while the smoke test is
running. That is expected. If a smoke job fails, its dependent full array will
not run.

The array-index mapping is deterministic:

```text
0-26   child
27-53  adolescent
54-80  adult

within a cohort:
index offset = (patient_number - 2) * 3 + policy_seed
```

The complete formula is:

```text
array_index = cohort_index * 27 + (patient_number - 2) * 3 + policy_seed
cohort_index: child=0, adolescent=1, adult=2
```

For example, `adolescent#005`, policy seed `2` is:

```text
27 + (5 - 2) * 3 + 2 = 38
```

This mapping is useful when only a few array cells fail.

### 4.6 Check result completeness

After both arrays finish, count all three required file types:

```bash
BASE_ROOT="author_v1_all_baselines_evaluation/no_shield/t1d/$ALGORITHM"
SHIELD_ROOT="author_v1_all_baselines_evaluation/predictive_penalty10/t1d/$ALGORITHM"

for root in "$BASE_ROOT" "$SHIELD_ROOT"; do
  printf '%s\n' "$root"
  printf '  summaries: '
  find "$root" -type f -name summary_results.csv | wc -l
  printf '  details:   '
  find "$root" -type f -name detailed_results.csv | wc -l
  printf '  metadata:  '
  find "$root" -type f -name run_metadata.txt | wc -l
done
```

Every printed count must be `81`. A scheduler state of `COMPLETED` is not a
substitute for checking result files.

### 4.7 Run the strict validator and aggregator

Create a small directory for human-readable validation logs:

```bash
mkdir -p validation_reports
```

Validate and aggregate no Shield:

```bash
python scripts/aggregate_author_v1_t1d.py \
  "$BASE_ROOT" \
  --algorithm "$ALGORITHM" \
  --condition no_shield \
  --expected-episodes 1 \
  --expected-result-collection author_v1_all_baselines_evaluation \
  | tee "validation_reports/${ALGORITHM}_no_shield.txt"
```

Validate and aggregate the predictive Shield:

```bash
python scripts/aggregate_author_v1_t1d.py \
  "$SHIELD_ROOT" \
  --algorithm "$ALGORITHM" \
  --condition predictive \
  --logit-penalty 10 \
  --expected-episodes 1 \
  --expected-result-collection author_v1_all_baselines_evaluation \
  | tee "validation_reports/${ALGORITHM}_predictive.txt"
```

Both commands must print:

```text
Matrix: 81/81 cells, 81 episode rows
Checkpoint identities: 9/9 cohort/seed pairs
Validation: PASS
```

The predictive command must additionally print:

```text
Predictor identities: 1/1
```

The validator checks file completeness, episode rows, episode numbering,
patient and seed labels, source commits, checkpoint epoch, condition,
result-collection name, predictor settings, policy hashes, and predictor hash.
It then prints cohort values for every training seed and final mean, sample SD,
and population SD for TIR, Risk Index, and CV.

Do not aggregate a condition that fails validation. Fix or rerun the bad cells
first.

### 4.8 Record the paired effect

For every metric, calculate:

```text
paired change = predictive-Shield mean - no-Shield mean
```

Interpret the signs as follows:

| Metric | Better direction | Desired paired sign |
|---|---|---:|
| TIR | higher | positive |
| Risk Index | lower | negative |
| CV | lower | negative |

A positive TIR change does not excuse a large increase in Risk Index or CV.
Report all three metrics together.

## 5. Numerical references and the paper revision

This workflow intentionally pins the January author-v1 source commits, so use
the January paper revision for the main comparison:

- [arXiv v1, submitted 28 January 2026](https://arxiv.org/html/2601.21094v1)
- [current arXiv record and revision history](https://arxiv.org/abs/2601.21094)

Do not silently compare an author-v1 run with a value copied from the May v2
paper. Record the paper revision and table number in every report.

The following reference is Table 3 of arXiv v1. “Implied no Shield” is computed
from the reported Shield mean and reported paired change; it is not a separate
quoted table entry.

### TIR reference

| Repository name | Paper label | Shield TIR | Paper change | Implied no Shield |
|---|---|---:|---:|---:|
| CPO | CPO | 85.50 ± 1.65 | +4.70 | 80.80 |
| CUP | CUP | 73.33 ± 4.08 | -0.22 | 73.55 |
| FOCOPS | FOCOPS | 82.63 ± 2.27 | +3.07 | 79.56 |
| OnCRPO | CRPO | 83.76 ± 2.67 | +7.83 | 75.93 |
| PCPO | PCPO | 35.12 ± 1.25 | -0.14 | 35.26 |
| PPOLag | PPOLag | 85.59 ± 0.99 | +8.05 | 77.54 |
| RCPO | RCPO | 86.45 ± 0.58 | +6.90 | 79.55 |
| TRPOLag | TRPOLag | 82.70 ± 2.16 | +5.79 | 76.91 |

### Risk Index reference

| Repository name | Shield Risk | Paper change | Implied no Shield |
|---|---:|---:|---:|
| CPO | 3.44 ± 0.31 | -1.51 | 4.95 |
| CUP | 7.14 ± 1.12 | -0.04 | 7.18 |
| FOCOPS | 4.53 ± 0.42 | -0.97 | 5.50 |
| OnCRPO | 3.92 ± 0.57 | -2.16 | 6.08 |
| PCPO | 21.30 ± 0.75 | +0.19 | 21.11 |
| PPOLag | 3.96 ± 0.22 | -1.79 | 5.75 |
| RCPO | 3.34 ± 0.04 | -2.26 | 5.60 |
| TRPOLag | 4.67 ± 0.67 | -1.35 | 6.02 |

### CV reference

| Repository name | Shield CV | Paper change | Implied no Shield |
|---|---:|---:|---:|
| CPO | 25.40 ± 1.08 | -2.56 | 27.96 |
| CUP | 35.93 ± 2.70 | -0.72 | 36.65 |
| FOCOPS | 29.63 ± 3.52 | -4.26 | 33.89 |
| OnCRPO | 28.77 ± 1.03 | -1.16 | 29.93 |
| PCPO | 46.40 ± 0.86 | -3.58 | 49.98 |
| PPOLag | 29.18 ± 0.16 | -3.21 | 32.39 |
| RCPO | 26.22 ± 1.24 | -3.92 | 30.14 |
| TRPOLag | 31.29 ± 0.74 | -0.35 | 31.64 |

The v1 appendix also contains T1D aggregate tables whose numbers are not
identical to the main paired table. Do not combine the main-table Shield value,
an appendix no-Shield value, and a newly reproduced delta in one comparison.
Choose one reference table, name it, and keep all arithmetic internally
consistent. For this author-v1 workflow, Table 3 is the primary reference
because the reconstructed CPO no-Shield result matches its implied baseline.

## 6. Known CPO canary result

The completed one-episode CPO matrix is a useful check that the shared setup is
still behaving as expected.

| Condition | TIR (%) | Risk Index | CV (%) |
|---|---:|---:|---:|
| no Shield | 80.8006 ± 3.3275 | 4.9569 ± 0.8490 | 27.9618 ± 0.9227 |
| author-v1 default predictive Shield | 83.5852 ± 3.7958 | 4.3509 ± 0.8634 | 27.7103 ± 1.2233 |
| paired change | +2.7846 | -0.6061 | -0.2515 |

These standard deviations are sample SDs across the three policy-training-seed
aggregates. The matching population SDs are also printed by the validator.

The no-Shield CPO mean matches Table 3's implied baseline almost exactly. The
predictive-Shield improvement is smaller than the paper reports. This is why
the remaining algorithms must report the baseline and Shield results
separately: a matching baseline does not prove that the Shield configuration or
predictor exactly matches the authors' unpublished final setup.

## 7. Evaluation-seed limitation

The evaluator records seed `22` and passes `22 + episode_index` to reset.
However, the January GlucoSim constructor creates a fixed JAX key from seed `0`,
and its reset path does not replace that existing key. Consequently:

- the policy RNG is seeded with `22`;
- no-Shield and Shield remain paired because each process begins from the same
  fixed JAX key;
- a multi-episode run advances sequential splits of that key;
- ten episodes must not be described as ten independent simulator seeds
  `22-31`.

For the initial one-episode validation this does not invalidate the pairing,
but it must be recorded in any methodological write-up.

## 8. Recovering from failures

### 8.1 Find the failed array indices

```bash
sacct -X -j <JOB_ID> \
  --format=JobID,JobName,State,ExitCode,Elapsed \
  --noheader
```

Look for array elements ending in `_N`, where `N` is the failed index. Read the
matching log before retrying:

```bash
less slurm-<JOB_NAME>-<JOB_ID>_<ARRAY_INDEX>.out
```

### 8.2 Retry only failed no-Shield cells

For example, to retry indices `17`, `38`, and `75`:

```bash
sbatch --array=17,38,75 \
  jobs/evaluate_author_v1_t1d_baseline.slurm \
  "$ALGORITHM" no_shield 1 10 author_v1_all_baselines_evaluation
```

### 8.3 Retry only failed predictive-Shield cells

```bash
sbatch --array=17,38,75 \
  jobs/evaluate_author_v1_t1d_baseline.slurm \
  "$ALGORITHM" predictive 1 10 author_v1_all_baselines_evaluation
```

After a retry, rerun the three file counts and the strict validator. Do not
assume that a successful retry repaired the entire matrix.

### 8.4 Common failure patterns

| Symptom | Likely cause | What to check |
|---|---|---|
| smoke fails before Python starts | missing archive, patch, checkpoint, or predictor | run the Section 4.2 preflight |
| predictive fails but no Shield passes | missing predictor/transitions or predictor load error | predictor hashes and 180 transition files |
| full array never starts | smoke dependency failed | `sacct` for the smoke job and its log |
| exactly 80 result cells | smoke result missing or overwritten | array cell `0` paths |
| only one cohort is missing | failed contiguous array range | index ranges in Section 4.5 |
| validator reports mixed hashes | files from different checkpoint downloads or runs | manifest and every `run_metadata.txt` |
| metrics differ greatly while metadata passes | paper revision, aggregation, stochastic episode, or unpublished Shield asset mismatch | compare baseline first, then Shield |

## 9. What each validator submits to the team

For every assigned algorithm, provide:

1. the four Slurm job IDs;
2. the checkpoint manifest revision/hashes;
3. the two saved validator outputs from `validation_reports/`;
4. a compact result table containing no Shield, predictive Shield, paired
   change, paper reference, and reproduction-minus-paper difference;
5. the number of early-terminated episodes printed by each validation run;
6. a short discrepancy note with no unsupported causal claim;
7. structural and numerical status labels.

Use this template:

```text
Algorithm (paper label):
Validator:
Date:
Repository commit:
Paper revision/table: arXiv v1 / Table 3

Slurm jobs:
  no-Shield smoke:
  no-Shield full:
  predictive smoke:
  predictive full:

Provenance:
  GlucoAlg commit: 50d3134fd6f0e01019295be0d6b99f110f794753
  GlucoSim commit: 6522d49ee1e6c1fd34f287519a5d41df735bbd38
  checkpoint epoch: 2441
  policy seeds: 0, 1, 2
  evaluation seed field: 22
  predictor SHA-256:

Completeness:
  no Shield: 81 summaries / 81 details / 81 metadata
  predictive: 81 summaries / 81 details / 81 metadata
  validator: PASS / FAIL

Results (mean ± sample SD):
  no Shield: TIR ..., Risk ..., CV ...
  predictive: TIR ..., Risk ..., CV ...
  paired change: TIR ..., Risk ..., CV ...

Paper comparison:
  reproduced minus paper Shield mean: ...
  reproduced minus paper paired change: ...

Status:
  structural:
  numerical:

Notes / unresolved discrepancies:
```

## 10. Final checklist

Before marking one algorithm complete, confirm every item:

- [ ] I used the exact repository algorithm name.
- [ ] I did not run compute work on the login node.
- [ ] All nine cohort/seed checkpoint pairs exist at epoch `2441`.
- [ ] The author-v1 archive hashes match Section 3.3.
- [ ] The group predictor hashes match Section 3.5.
- [ ] Both smoke tests completed with exit code `0:0`.
- [ ] Both full arrays completed, including the smoke cell.
- [ ] Both conditions have 81 summaries, 81 details, and 81 metadata files.
- [ ] Both strict validator commands print `Validation: PASS`.
- [ ] I saved the validator output under `validation_reports/`.
- [ ] I reported TIR, Risk Index, and CV together.
- [ ] I computed the paired Shield-minus-baseline change.
- [ ] I named the paper revision and table used for comparison.
- [ ] I reported exact numerical differences without inventing a cause.
- [ ] I did not describe multi-episode runs as independent seeds `22-31`.
- [ ] I did not add raw results, checkpoints, archives, or Slurm logs to Git.

## 11. Files that should remain outside Git

Do not commit generated data or large binary artifacts:

```text
author_v1_GlucoAlg_50d3134.tar.gz
author_v1_GlucoSim_6522d49.tar.gz
glucoalg_checkpoints_*.zip
author_v1_evaluation/
author_v1_all_baselines_evaluation/
cpo_full_evaluation/
v1_baseline_evaluation/
slurm-*.out
```

Commit only the small scripts, patches, documentation, and compact validation
reports that the team has explicitly agreed to version. Always inspect
`git status --short` before staging files; do not use `git add .` in a checkout
that contains raw evaluation outputs.
