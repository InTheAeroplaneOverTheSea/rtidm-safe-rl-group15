# Code handoff: paper-aligned predictive Shield

## Version layout

The parent repository pins the January upstream GlucoAlg/GlucoSim revisions.
The group's reviewed submodule commits are distributed as small incremental Git
bundles under `vendor/`. After a fresh clone, run:

```bash
git submodule update --init --recursive
bash scripts/apply_upstream_patches.sh
```

The script refuses to modify dirty submodules or unexpected upstream revisions,
fetches the reviewed commits from the bundles, and checks out those exact
commits in detached-HEAD mode.
The release branch is:

```text
codex/paper-aligned-predictive-shield
```

The implementation now:

- preserves the public CPO policy architecture and the trained legacy BA-NODE
  checkpoint format;
- implements the published `<60`, `60--80`, and `70--180 mg/dL` gating/static
  rules;
- evaluates top-k bolus levels against every meal level and masks unsafe
  bolus-meal pairs jointly instead of masking both marginals;
- requires an explicit predictive upper threshold below `180 mg/dL`;
- restricts the logit-penalty magnitude to the paper's `{5, 10, 15}` grid;
- reconstructs online BA-NODE context using executed actions and cumulative BG
  deltas, matching the legacy training tensors;
- records the predictive decision trace, thresholds, seeds, perturbation, and
  clinically named excursion metrics with each evaluation;
- saves a real best-validation BA-NODE checkpoint and real periodic epoch
  checkpoints during future training runs.

## What is deliberately not claimed

The paper does not publish the selected upper predictive threshold, and this
repository has no author-supplied final Shield configuration. `170 mg/dL` is
therefore only the current smoke-test value. The defaults `top-k=2` and
`context_examples=20` are also implementation choices that must be reported.

Passing unit tests or one paired episode proves that the code path runs and the
Shield changes the action distribution. It does not prove the paper's safety
effect. A final claim requires a frozen validation choice followed by paired,
multi-episode evaluation on held-out policy seeds and patients.

## Compatibility and data

The following large files are intentionally excluded from Git:

```text
external/GlucoAlg/saved_files/env_transitions/**/*.npz
external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/
external/GlucoAlg/saved_models/t1d/adolescent/CPO/seed{0,1,2}/
external/GlucoAlg/diabetes_evaluation/
```

Required CPO files for each seed are `config.json` and
`torch_save/epoch-2441.pt`. The BA-NODE folder must contain `config.yaml` and
`best_model.pth`; `model.pth` is useful as the final-epoch fallback. The setup
job downloads 180 transition `.npz` files from the public dataset repository.

Old CPO and BA-NODE checkpoints still load because no network dimensions or
state-dict keys were changed. This is runtime compatibility, not proof that a
legacy predictor was trained with a causally ideal data collector. The legacy
collector stores post-step observations with the preceding action; the online
context code mirrors that convention so the existing checkpoint is usable.

## Verification performed on 2026-09-29

The test suite passed:

```text
external/GlucoAlg tests: 18 passed
project perturbation tests: 5 passed
```

The actual CPO seed-0 epoch-2441 checkpoint and the seed-42 BA-NODE
`best_model.pth` loaded together. A one-episode, four-condition smoke test used
`adolescent#010`, evaluation seed `22`, upper threshold `170`, penalty `10`,
top-k `2`, and 20 context examples.

| Condition | TIR | Time >180 | Risk index | Result |
|---|---:|---:|---:|---|
| clean, no Shield | 85.962% | 14.038% | 4.8182 | baseline |
| clean, predictive Shield | 86.012% | 13.988% | 4.8108 | small improvement |
| +30, no Shield | 85.962% | 14.038% | 4.8182 | baseline |
| +30, predictive Shield | 85.962% | 14.038% | 4.8182 | no physiological change in this episode |

After fixing the context interface, predicted trajectories were in plausible
ranges: approximately `111--214 mg/dL` for clean and `143--238 mg/dL` for the
`+30 mg/dL` controller bias. In the perturbed episode, modified recommendations
were rejected by the simulator's acceptance windows, so identical physiological
metrics are expected. Treat this only as a functional smoke test.

## Where results are written

For the smoke configuration, outputs are under:

```text
external/GlucoAlg/diabetes_evaluation/t1d/CPO/adolescent#010/seed0/
  no_shield/
  no_shield_cgm_bias_plus30/
  with_shield_10_hyper170/
  with_shield_10_hyper170_cgm_bias_plus30/
```

Each condition contains `evaluation_config.json`, `perturbation.json`,
`detailed_results.csv`, `summary_results.csv`, and per-step controller traces.
The trace fields beginning with `shield_` show when predictive evaluation was
ready, which pairs were unsafe, and the predicted BG range.

## Recommended next experiment

1. Reserve validation patients/evaluation seeds and select the upper threshold
   and penalty only on that split.
2. Freeze the chosen values.
3. Run clean no-Shield versus clean predictive-Shield on CPO seeds 0, 1, and 2
   over held-out patients with multiple episodes.
4. Then run the same paired comparison under `cgm_bias_plus30`.
5. Report paired changes in time below 70, time above 180/250, excursion counts,
   TIR, risk index, and intervention/unsafe-pair counts.

Do not select the best threshold on the same test patients used for the final
reported result.
