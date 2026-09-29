# RTiDM Safe RL Group 15

Research Topics in Data Mining  
Safe Reinforcement Learning  
Group 15

## Project

This project investigates the robustness and safety of reinforcement learning methods for blood glucose control under broader real-world stress conditions, including noise, rare events, and distribution shifts.

## Upstream Code

This project builds upon:

- GlucoSim
- GlucoAlg

Both are included as Git submodules under `external/`.

## Repository Structure

- `external/`: upstream research code
- `src/`: our implementation
- `experiments/`: experiment scripts
- `configs/`: experiment configurations
- `results/`: processed results
- `docs/`: project documentation

## Setup

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/InTheAeroplaneOverTheSea/rtidm-safe-rl-group15.git
cd rtidm-safe-rl-group15
bash scripts/apply_upstream_patches.sh
```

Checkpoints, transition datasets, and evaluation output are not stored in Git.
The setup command restores the group's reviewed GlucoAlg and GlucoSim commits on
top of the pinned upstream revisions.

## Evaluation perturbations

Evaluation-only observation perturbations live under
`src/rtidm_safe_rl/perturbations/` and are selected with YAML files in
`configs/perturbations/`. This keeps experimental interventions separate from
GlucoSim physiology and from the GlucoAlg evaluation implementation.

Run a clean condition without Shield:

```bash
python experiments/evaluate_with_perturbations.py \
  t1d CPO adolescent#010 0 --epoch 2441 --num-episodes 1 \
  --shield-type none \
  --perturbation-config configs/perturbations/none.yaml
```

Predictive Shield runs require an explicitly calibrated upper threshold below
180 mg/dL. The following `170` value is a smoke-test configuration, not a
claimed paper hyperparameter:

```bash
python experiments/evaluate_with_perturbations.py \
  t1d CPO adolescent#010 0 --epoch 2441 --num-episodes 1 \
  --shield --shield-type predictive \
  --dynamics-model-dir saved_files/dynamics_predictor/ba_node_b5_p24_h24/42 \
  --hyper-check-threshold 170 --logit-penalty 10 \
  --perturbation-config configs/perturbations/cgm_bias_plus30.yaml
```

See `docs/CODE_HANDOFF.md` for implementation status and unresolved research
choices, `docs/SURF_QUICKSTART.md` for the shortest cluster workflow, and
`docs/perturbations.md` for the perturbation format.
