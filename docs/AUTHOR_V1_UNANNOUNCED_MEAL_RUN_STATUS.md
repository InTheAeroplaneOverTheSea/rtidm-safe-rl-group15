# Author-v1 unexpected-meal SURF run status

Last updated: 2026-10-11 22:30 CEST

This file records the submission and the first post-submission Slurm snapshot.
It is not a live status page. Automatic polling was intentionally stopped after
Slurm confirmed that both smoke jobs were running and both full arrays had the
expected dependencies. A later one-shot snapshot is recorded below.

> **Superseded as a direct comparator:**
> `author_v1_unannounced_meal_evaluation` contains only the hidden-meal
> scenario. It is retained as a full 81-cell pipeline gate, but it must not be
> subtracted directly from `author_v1_all_baselines_evaluation` because the
> stress harness uses corrected episode-key reset semantics. The formal
> replacement reruns matched nominal and hidden-meal scenarios together with
> ten independent seeds per cell in
> `author_v1_unannounced_meal_validation_bridge_10seed`.

## Submitted comparison

| Field | Value |
| --- | --- |
| Algorithms | `CPO`, `CUP`, `PCPO`, `PPOLag` (Person A/C allocation) |
| Scenario | `unannounced_meal_60g` |
| Arms | no Shield; predictive Shield |
| Predictive meal-hyper check | `false` |
| Hidden meal | `60 g` daily at `15:00`, injected post-action/pre-step |
| Episodes per cell | `1` |
| Replicate id | `0` |
| Predictive logit penalty | `10` |
| Matrix | `81` patient/checkpoint cells per arm |
| Result collection | `author_v1_unannounced_meal_evaluation` |
| Remote account | `scur0316@snellius.surf.nl` |
| Remote project | `/home/scur0316/projects/rtidm-safe-rl-group15` |

## Original CPO Slurm jobs

| Job | Job ID | Initial snapshot |
| --- | ---: | --- |
| no-Shield smoke, array cell `0` | `27888674` | `RUNNING` on `gcn50` |
| predictive-Shield smoke, array cell `0` | `27888675` | `RUNNING` on `gcn50` |
| no-Shield full, array cells `1-80` | `27888677` | `PENDING (Dependency)` |
| predictive-Shield full, array cells `1-80` | `27888678` | `PENDING (Dependency)` |

The two full arrays start only after their corresponding smoke job succeeds.
`PENDING (Dependency)` was therefore the expected initial state, not an error.

## Additional Person A/C submissions

No queue or accounting query was made after these submissions. The table only
records the IDs returned by `sbatch` on 2026-10-11 at 21:24 CEST.

The shared-checkout checkpoint manifest initially contained only CPO. Per the
shared setup instructions, one staging job was submitted to download and
verify all eight pinned January algorithm checkpoints:

| Job | Job ID | Dependency |
| --- | ---: | --- |
| all-algorithm checkpoint staging | `27894983` | none |

Both smoke jobs for each new algorithm have `afterok:27894983`; their full
arrays retain the usual `afterok` dependency on the corresponding smoke job.

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CUP` | `27895033` | `27895034` | `27895035` | `27895036` |
| `PCPO` | `27895037` | `27895038` | `27895039` | `27895040` |
| `PPOLag` | `27895041` | `27895048` | `27895049` | `27895050` |

The durable submission record on SURF is:

```text
author_v1_unannounced_meal_evaluation/
  unannounced_meal_60g/replicate0/submitted_jobs.tsv
  unannounced_meal_60g/replicate0/prerequisite_jobs.tsv
```

## One-shot manual check

Run this only when a new snapshot is wanted:

```bash
ssh -i ~/.ssh/id_ed25519_surf scur0316@snellius.surf.nl
cd ~/projects/rtidm-safe-rl-group15
bash scripts/check_author_v1_unannounced_meal_person_ac_status.sh
```

The combined checker performs one `squeue` query and one `sacct` query for the
checkpoint prerequisite plus all Person A/C evaluation jobs, then exits. It
never sleeps or polls. It checks result completeness for all four algorithms;
for each algorithm whose two arms both contain all 81 result cells, it also
runs the no-Shield aggregation, predictive aggregation, and paired-disturbance
validator and saves their reports under `validation_reports/`.

## One-shot snapshot at 21:49 CEST

The checkpoint staging job `27894983` completed with exit code `0:0`. The
one-shot checker reported:

| Algorithm | no-Shield files | predictive files | Validation |
| --- | ---: | ---: | --- |
| `CPO` | `81/81` | `81/81` | both aggregations PASS; pairing PASS |
| `CUP` | `81/81` | `1/81` | predictive still running |
| `PCPO` | `81/81` | `3/81` | predictive still running |
| `PPOLag` | `81/81` | `1/81` | predictive still running |

The completed CPO hidden-meal gate produced:

| Condition | TIR (%) | Risk Index | CV (%) | Early termination |
| --- | ---: | ---: | ---: | ---: |
| no Shield | `64.6187 +/- 2.1190` | `11.7709 +/- 1.1069` | `51.0350 +/- 2.2944` | `36/81` |
| predictive | `65.4800 +/- 1.5122` | `11.2430 +/- 0.6091` | `49.9360 +/- 1.4164` | `29/81` |

These values are gate diagnostics, not the unexpected-meal effect. The meal
effect requires the matched nominal scenario from the formal bridge.

## Canceled one-episode bridge submission

Submitted on 2026-10-11 at 21:51--21:52 CEST. No `squeue` or `sacct` query was
made after this submission. The tables below only report the job IDs returned
by `sbatch` and written to the durable submission records.

This submission used the wrong one-episode protocol. All of its remaining jobs
were canceled at 22:07 CEST, and a one-shot confirmation found zero remaining
queue rows. Its partial outputs must not be used for the formal comparison.

Both scenarios use all eight algorithms, one episode per cell, replicate `0`,
predictive logit penalty `10`, and the same `81` patient/checkpoint cells per
arm. Crucially, **both matched nominal and hidden `60 g` have a no-Shield arm
and a predictive-Shield arm**.

Nominal jobs:

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CPO` | `27895909` | `27895910` | `27895911` | `27895912` |
| `CUP` | `27895914` | `27895915` | `27895916` | `27895917` |
| `FOCOPS` | `27895918` | `27895919` | `27895920` | `27895921` |
| `OnCRPO` | `27895922` | `27895923` | `27895924` | `27895925` |
| `PCPO` | `27895926` | `27895929` | `27895930` | `27895932` |
| `PPOLag` | `27895933` | `27895934` | `27895935` | `27895936` |
| `RCPO` | `27895937` | `27895938` | `27895939` | `27895940` |
| `TRPOLag` | `27895941` | `27895943` | `27895944` | `27895947` |

Hidden `60 g` jobs:

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CPO` | `27895948` | `27895949` | `27895950` | `27895951` |
| `CUP` | `27895952` | `27895953` | `27895954` | `27895955` |
| `FOCOPS` | `27895956` | `27895957` | `27895958` | `27895959` |
| `OnCRPO` | `27895960` | `27895961` | `27895962` | `27895963` |
| `PCPO` | `27895964` | `27895965` | `27895966` | `27895967` |
| `PPOLag` | `27895968` | `27895969` | `27895975` | `27895977` |
| `RCPO` | `27895979` | `27895981` | `27895982` | `27895983` |
| `TRPOLag` | `27895984` | `27895985` | `27895986` | `27895987` |

The records were checked structurally without polling Slurm: each file has
exactly `32` data rows, and each of the eight algorithms has exactly the four
expected condition/phase entries. They are stored at:

```text
author_v1_unannounced_meal_validation_bridge/
  nominal/replicate0/submitted_jobs.tsv
  unannounced_meal_60g/replicate0/submitted_jobs.tsv
```

The records are retained only as a cancellation audit. Do not run the formal
bridge checker against this collection.

## Correct ten-seed formal validation bridge (canceled)

Submitted on 2026-10-11 at 22:10--22:12 CEST into the fresh collection
`author_v1_unannounced_meal_validation_bridge_10seed`.

At 22:29 CEST, the user requested cancellation so that the Person A/C
one-seed bridge could receive resources first. `scancel` was issued for all 64
parent IDs. A post-cancellation snapshot found no pending or running ten-seed
jobs and only one task in `COMPLETING`. Partial results already synchronized to
the collection remain on disk; resuming this experiment requires identifying
and resubmitting missing array indices rather than rerunning complete cells.

| Field | Value |
| --- | --- |
| Algorithms | all eight author-v1 algorithms |
| Scenarios | `nominal`; `unannounced_meal_60g` |
| Arms per scenario | no Shield; predictive Shield |
| Patient/checkpoint cells per arm | `81` |
| Independent evaluation seeds per cell | `22..31` (`10`) |
| Total episode trajectories | `25,920` |
| Replicate id | `0` |
| Predictive logit penalty | `10` |
| Slurm time limit per cell | `02:00:00` |
| Full-array throttle | no Shield `%18`; predictive `%18` |

Nominal jobs:

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CPO` | `27896931` | `27896932` | `27896934` | `27896936` |
| `CUP` | `27896940` | `27896942` | `27896944` | `27896947` |
| `FOCOPS` | `27896949` | `27896951` | `27896953` | `27896955` |
| `OnCRPO` | `27896958` | `27896959` | `27896961` | `27896962` |
| `PCPO` | `27896965` | `27896967` | `27896969` | `27896970` |
| `PPOLag` | `27896974` | `27896975` | `27896977` | `27896979` |
| `RCPO` | `27896982` | `27896984` | `27896985` | `27896987` |
| `TRPOLag` | `27896989` | `27896991` | `27896993` | `27896995` |

Hidden `60 g` jobs:

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CPO` | `27896998` | `27897000` | `27897002` | `27897003` |
| `CUP` | `27897006` | `27897007` | `27897009` | `27897011` |
| `FOCOPS` | `27897014` | `27897015` | `27897018` | `27897020` |
| `OnCRPO` | `27897023` | `27897024` | `27897026` | `27897028` |
| `PCPO` | `27897032` | `27897034` | `27897036` | `27897038` |
| `PPOLag` | `27897041` | `27897042` | `27897044` | `27897047` |
| `RCPO` | `27897050` | `27897051` | `27897054` | `27897056` |
| `TRPOLag` | `27897059` | `27897060` | `27897062` | `27897068` |

Static verification, without a progress poll, confirmed that both submission
records contain exactly `32` rows, every algorithm has its four expected jobs,
every row records `episodes=10`, all `64` jobs have `TimeLimit=02:00:00`, and
all `16` predictive full arrays have `ArrayTaskThrottle=18`.

When a one-shot result snapshot is wanted, run:

```bash
bash scripts/check_author_v1_unannounced_meal_validation_bridge_status.sh
```

The checker defaults to the ten-seed collection, makes one queue query and one
accounting query, validates any complete arms with `--expected-episodes 10`,
and exits. It does not poll.

## Person A/C one-seed bridge

At 22:21--22:23 CEST, a separate fast bridge was submitted for the Person A/C
algorithms only: `CPO`, `CUP`, `PCPO`, and `PPOLag`. Its fresh collection is
`author_v1_unannounced_meal_validation_bridge_1seed`.

For replicate `0`, the current remote job sets `EVALUATION_SEED_BASE=22`; the
stress hook computes `base + replicate * 10000 + episode`. Thus the single
episode in every cell uses literal simulator seed `22`. Policy-training seeds
`0`, `1`, and `2` remain the separate checkpoint dimension of the 81-cell
matrix.

Nominal jobs:

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CPO` | `27898125` | `27898127` | `27898143` | `27898146` |
| `CUP` | `27898150` | `27898154` | `27898157` | `27898160` |
| `PCPO` | `27898184` | `27898186` | `27898188` | `27898190` |
| `PPOLag` | `27898193` | `27898195` | `27898196` | `27898198` |

Hidden `60 g` jobs:

| Algorithm | no-Shield smoke | predictive smoke | no-Shield full | predictive full |
| --- | ---: | ---: | ---: | ---: |
| `CPO` | `27898217` | `27898219` | `27898221` | `27898232` |
| `CUP` | `27898234` | `27898236` | `27898257` | `27898259` |
| `PCPO` | `27898282` | `27898290` | `27898292` | `27898295` |
| `PPOLag` | `27898298` | `27898305` | `27898306` | `27898308` |

The interrupted all-algorithm command also created `32` out-of-scope parent
jobs. They were canceled, are absent from the queue, and are retained only in
`canceled_out_of_scope_jobs.tsv`; both active submission records now contain
exactly `16` rows with `episodes=1` and the four Person A/C algorithms.

An attempt to hold pending ten-seed array elements was rejected by the
Snellius controller with `Access/permission denied`. The user subsequently
authorized full cancellation of the remaining ten-seed jobs, as recorded in
the preceding section. The one-seed batch therefore no longer competes with
the ten-seed batch for new allocations.

## Interpretation boundary

The earlier local two-episode CPO check established that the workflow executes,
the two arms share literal episode seeds and the same seven-day hidden-meal
plan, both arms receive at least one hidden meal, and predictive metadata
records `v1_use_meal_hyper_check=false`. It was only a one-cell smoke test and
is not a performance conclusion. Clinical metric comparisons should wait for
the full SURF matrices and all three validators to pass.
