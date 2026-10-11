# Author-v1 unexpected-meal SURF run status

Last updated: 2026-10-11 17:11 CEST

This file records the submission and the first post-submission Slurm snapshot.
It is not a live status page. Automatic polling was intentionally stopped after
Slurm confirmed that both smoke jobs were running and both full arrays had the
expected dependencies.

## Submitted comparison

| Field | Value |
| --- | --- |
| Algorithm | `CPO` |
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

## Slurm jobs

| Job | Job ID | Initial snapshot |
| --- | ---: | --- |
| no-Shield smoke, array cell `0` | `27888674` | `RUNNING` on `gcn50` |
| predictive-Shield smoke, array cell `0` | `27888675` | `RUNNING` on `gcn50` |
| no-Shield full, array cells `1-80` | `27888677` | `PENDING (Dependency)` |
| predictive-Shield full, array cells `1-80` | `27888678` | `PENDING (Dependency)` |

The two full arrays start only after their corresponding smoke job succeeds.
`PENDING (Dependency)` was therefore the expected initial state, not an error.

The durable submission record on SURF is:

```text
author_v1_unannounced_meal_evaluation/
  unannounced_meal_60g/replicate0/submitted_jobs.tsv
```

## One-shot manual check

Run this only when a new snapshot is wanted:

```bash
ssh -i ~/.ssh/id_ed25519_surf scur0316@snellius.surf.nl
cd ~/projects/rtidm-safe-rl-group15
bash scripts/check_author_v1_unannounced_meal_status.sh
```

The checker performs one `squeue` query, one `sacct` query, and one result-count
pass, then exits. It never loops or sleeps. If both arms contain all 81 result
cells, it also runs the no-Shield aggregation, predictive aggregation, and
paired-disturbance validator and saves their reports under
`validation_reports/`.

## Interpretation boundary

The earlier local two-episode CPO check established that the workflow executes,
the two arms share literal episode seeds and the same seven-day hidden-meal
plan, both arms receive at least one hidden meal, and predictive metadata
records `v1_use_meal_hyper_check=false`. It was only a one-cell smoke test and
is not a performance conclusion. Clinical metric comparisons should wait for
the full SURF matrices and all three validators to pass.
