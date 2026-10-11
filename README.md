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
```

## Validation guide

Team members validating the eight author-v1 T1D baselines should follow the
step-by-step [`3 + 3 + 2` validation tutorial](docs/AUTHOR_V1_T1D_VALIDATION_TUTORIAL.md).
