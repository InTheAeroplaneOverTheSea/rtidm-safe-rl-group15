# SURF quick start

This is the short operational guide. See `surf_training.md` for the detailed
explanations and `CODE_HANDOFF.md` for scientific limitations.

## 1. Upload from macOS

Apple's bundled `rsync` does not support `--info=progress2`, so use `--progress`:

```bash
cd /Users/yulema/Developer/rtidm-safe-rl-group15
rsync -rlptz --progress \
  -e 'ssh -i /Users/yulema/.ssh/id_ed25519_surf -o IdentitiesOnly=yes' \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  ./ scur0316@snellius.surf.nl:~/projects/rtidm-safe-rl-group15/
```

## 2. Log in and set up the environment

```bash
ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519_surf" \
  scur0316@snellius.surf.nl

cd ~/projects/rtidm-safe-rl-group15
sbatch jobs/setup_surf.slurm
```

Check the returned job ID:

```bash
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,Elapsed,ExitCode
tail -n 100 slurm-rtidm-setup-<JOB_ID>.out
```

Success means `COMPLETED`, exit code `0:0`, and `Transition files: 180`.

## 3. Confirm required checkpoints

```bash
cd ~/projects/rtidm-safe-rl-group15

test -f external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/best_model.pth

for seed in 0 1 2; do
  test -f "external/GlucoAlg/saved_models/t1d/adolescent/CPO/seed${seed}/config.json"
  test -f "external/GlucoAlg/saved_models/t1d/adolescent/CPO/seed${seed}/torch_save/epoch-2441.pt"
done

find external/GlucoAlg/saved_files/env_transitions -name '*.npz' | wc -l
```

The final command must print `180`. Copy missing CPO or BA-NODE files from the
Mac with `rsync --progress`; do not commit checkpoints to Git.

## 4. Optional: retrain BA-NODE

One-epoch smoke test:

```bash
sbatch jobs/train_banode.slurm 1 999
```

Formal 100-epoch seed-42 run:

```bash
sbatch --job-name=banode-train --time=03:00:00 \
  jobs/train_banode.slurm 100 42
```

Use `best_model.pth` for downstream evaluation. Periodic
`model_epoch100.pth`, `model_epoch200.pth`, and so on are now saved during the
corresponding epoch rather than copied after training.

## 5. Four-condition Shield smoke test

This single submission runs clean and `+30 mg/dL`, each without and with the
predictive Shield:

```bash
sbatch --time=02:00:00 \
  jobs/evaluate_shield.slurm \
  1 all 0 'adolescent#010' 170 10 22
```

Arguments are:

```text
episodes perturbation policy_seed patient upper_threshold penalty eval_seed
```

`170` is a smoke-test threshold only. The script rejects a missing threshold,
a threshold outside `(0, 180)`, or a penalty outside `{5, 10, 15}`.

Monitor with the job ID printed by `sbatch`:

```bash
squeue -j <JOB_ID>
tail -f slurm-shield-eval-<JOB_ID>.out
```

If `squeue` is empty, the job has already finished; use:

```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,AllocTRES
tail -n 200 slurm-shield-eval-<JOB_ID>.out
```

One four-condition episode took roughly six minutes in the earlier SURF smoke
run. Ten episodes can take around one hour; actual queue and node times vary.

## 6. Multi-seed evaluation after calibration

After freezing an upper threshold and penalty on a validation split, run policy
seeds 0--2 as a Slurm array. The array ID becomes the policy seed:

```bash
sbatch --array=0-2 --time=02:00:00 \
  jobs/evaluate_shield.slurm \
  10 all 0 'adolescent#010' <THRESHOLD> <PENALTY> 22
```

Repeat for held-out patients. Do not call this a paper reproduction unless the
validation protocol and final multi-seed statistics are reported.

## 7. Download evaluation output

Run on the Mac:

```bash
rsync -rlptz --progress \
  -e 'ssh -i /Users/yulema/.ssh/id_ed25519_surf -o IdentitiesOnly=yes' \
  'scur0316@snellius.surf.nl:~/projects/rtidm-safe-rl-group15/external/GlucoAlg/diabetes_evaluation/' \
  '/Users/yulema/Developer/rtidm-safe-rl-group15/external/GlucoAlg/diabetes_evaluation/'
```
