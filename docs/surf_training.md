# SURF Snellius BA-NODE training guide

This guide prepares the GlucoAlg BA-NODE dynamics predictor on SURF Snellius,
runs a one-epoch GPU smoke test, and then runs the paper-aligned 100-epoch
training job. Commands are separated by where they must be executed.

## Account and paths

- Snellius login: `scur0316`
- Host: `snellius.surf.nl`
- Local repository:
  `/Users/yulema/Developer/rtidm-safe-rl-group15`
- Remote repository:
  `/home/scur0316/projects/rtidm-safe-rl-group15`
- Remote Python environment: `/home/scur0316/venvs/rtidm`

The login node is only for file transfer, preparing and submitting jobs, and
checking job status. Do not run training, Python tests, package installation,
or other compute work directly on the login node.

## 1. Connect from the Mac

Use the dedicated private key:

```bash
ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519_surf" \
  scur0316@snellius.surf.nl
```

The official Snellius ED25519 host fingerprint is:

```text
SHA256:2Vy9858ldWu3Xjt1a58MbhD5CjLIh1LCb8n/up0izGw
```

After login, verify the account and create the remote directories:

```bash
whoami
hostname
mkdir -p ~/projects/rtidm-safe-rl-group15 ~/venvs
```

`whoami` must print `scur0316`. The host name normally begins with `int`, which
identifies a login node.

## 2. Check account, GPU budget, partition, and quota

These are lightweight administrative commands and are appropriate on a login
node:

```bash
accinfo --product=gpu
budget-overview
myquota
sinfo -p staging,gpu_course -o "%P %a %l %D %c %m %G"
scontrol show partition gpu_course
```

Do not submit the GPU job if the account has no GPU budget or if
`gpu_course` is unavailable. `gpu_course` is course-specific and is not listed
in the public Snellius partition table; its use here is based on the partition
offered to this account by Open OnDemand. Follow course staff instructions if
they differ.

Exit back to the Mac before uploading:

```bash
exit
```

## 3. Upload the repository from the Mac

Run this in a local Mac terminal, not in the SSH session:

```bash
rsync -rlptz --progress \
  -e 'ssh -i /Users/yulema/.ssh/id_ed25519_surf -o IdentitiesOnly=yes' \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  /Users/yulema/Developer/rtidm-safe-rl-group15/ \
  scur0316@snellius.surf.nl:~/projects/rtidm-safe-rl-group15/
```

The command deliberately omits the macOS virtual environment and Git metadata.
The Linux environment is created separately on Snellius.

## 4. Submit environment setup to the staging partition

Reconnect:

```bash
ssh -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519_surf" \
  scur0316@snellius.surf.nl
```

Submit the setup job from the repository root:

```bash
cd ~/projects/rtidm-safe-rl-group15
sbatch jobs/setup_surf.slurm
```

Do not execute `jobs/setup_surf.sh` directly. It contains a guard that refuses
to run outside a Slurm allocation.

The setup job requests the `staging` partition with one CPU, 7 GiB memory, and
one hour. It:

1. loads the 2025 software stack;
2. loads `Python-bundle-PyPI/2025.04-GCCcore-14.2.0` and `CUDA/12.9.1`;
3. creates `~/venvs/rtidm`;
4. installs PyTorch and the minimal predictor-training dependencies;
5. downloads the public transition dataset from Hugging Face;
6. requires exactly 180 `.npz` transition files;
7. runs `pip check` and records the installed environment.

`torch.cuda.is_available()` may be false during this setup job because
`staging` is a CPU service partition. The GPU training job performs a separate
mandatory CUDA check.

## 5. Monitor the setup job

`sbatch` prints a job ID. Substitute it for `<SETUP_JOB_ID>` below:

```bash
squeue -j <SETUP_JOB_ID>
tail -f slurm-rtidm-setup-<SETUP_JOB_ID>.out
```

Pressing `Ctrl+C` stops only `tail`; it does not cancel the job. After the job
leaves the queue, inspect the recorded result:

```bash
sacct -j <SETUP_JOB_ID> --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

The expected result is `COMPLETED` with exit code `0:0`. The log should finish
with:

```text
SURF training environment is ready.
Transition files: 180
```

Lightweight checks on the login node are then sufficient:

```bash
test -f ~/venvs/rtidm/bin/activate && echo "venv present"
find external/GlucoAlg/saved_files/env_transitions -type f -name '*.npz' | wc -l
```

The second command must print `180`.

## 6. Submit the one-epoch GPU smoke test

Always submit from the repository root:

```bash
cd ~/projects/rtidm-safe-rl-group15
sbatch jobs/train_banode.slurm 1 999
```

The two arguments are `epochs=1` and `seed=999`. Seed 999 keeps the smoke-test
checkpoint separate from the formal seed-42 checkpoint.

The job requests:

- partition `gpu_course`;
- one node and one task;
- one `A100 1g.5gb` MIG GPU slice;
- two CPU cores;
- 16 GiB host memory;
- one hour wall time.

These course-specific values are derived from the live partition configuration:
each course node exposes 28 MIG slices, 72 CPUs, and approximately 480 GiB host
memory. Two CPUs and 16 GiB per slice allow all 28 slices to be scheduled without
the CPU or memory request becoming the limiting resource. This differs from the
public full-A100 example, which uses 18 CPUs for one complete A100 GPU.

At job start, the script:

1. loads the exact 2025 Python and CUDA modules;
2. activates the shared virtual environment;
3. copies GlucoAlg and its small input dataset into Slurm-provided `$TMPDIR`;
4. verifies all 180 transition files;
5. refuses to continue unless CUDA is available;
6. trains BA-NODE in `$TMPDIR`;
7. copies predictor checkpoints and logs back to the repository in home.

The script never uses `/tmp`; SURF reserves that path for system processes.

## 7. Monitor and verify the smoke test

Replace `<SMOKE_JOB_ID>` with the returned job ID:

```bash
squeue -j <SMOKE_JOB_ID>
tail -f slurm-banode-smoke-<SMOKE_JOB_ID>.out
```

The beginning of the log must contain:

```text
CUDA available: True
GPU: ...
GPU memory: approximately 5 GiB
Training BA-NODE: epochs=1, seed=999
```

If CUDA is false, the script exits instead of silently training on CPU.

After completion:

```bash
sacct -j <SMOKE_JOB_ID> \
  --format=JobID,JobName,Partition,State,Elapsed,AllocTRES,ExitCode

ls -lh \
  external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/999/
```

Expected output files include:

```text
config.yaml
best_model.pth
model.pth
logs/
```

## 8. Choose the formal wall time

Use the smoke test's elapsed time to estimate the 100-epoch job. SURF recommends
timing a minimal version first and requesting approximately 1.5 to 2 times the
estimated runtime. Do not blindly request the maximum wall time.

A conservative first estimate is:

```text
estimated runtime = 100 × one-epoch elapsed time
requested wall time = 1.5–2 × estimated runtime
```

The one-hour smoke job benefits from Snellius short-job scheduling. Ask for a
longer wall time only for the formal run.

## 9. Submit the paper-aligned 100-epoch training job

After choosing `HH:MM:SS`, submit:

```bash
cd ~/projects/rtidm-safe-rl-group15
sbatch \
  --job-name=banode-train \
  --time=HH:MM:SS \
  jobs/train_banode.slurm 100 42
```

Command-line Slurm options override the one-hour smoke-test defaults in the
script. This run uses the paper-aligned 100-epoch budget and formal seed 42.
During training, validation is evaluated every epoch; the downstream predictive
shield should load `best_model.pth`, the checkpoint with the lowest validation
error, rather than assuming epoch 100 is best.

Monitor it with:

```bash
squeue -u "$USER"
tail -f slurm-banode-train-<TRAIN_JOB_ID>.out
```

Cancel a wrong or wasteful job immediately with:

```bash
scancel <TRAIN_JOB_ID>
```

## 10. Verify and retrieve the formal predictor

On Snellius:

```bash
sacct -j <TRAIN_JOB_ID> \
  --format=JobID,JobName,Partition,State,Elapsed,AllocTRES,ExitCode

ls -lh \
  external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/
```

The formal directory must contain at least `config.yaml`, `best_model.pth`, and
`model.pth`.

To download the predictor, run this on the Mac:

```bash
mkdir -p \
  /Users/yulema/Developer/rtidm-safe-rl-group15/external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42

rsync -rlptz --progress \
  -e 'ssh -i /Users/yulema/.ssh/id_ed25519_surf -o IdentitiesOnly=yes' \
  scur0316@snellius.surf.nl:~/projects/rtidm-safe-rl-group15/external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/ \
  /Users/yulema/Developer/rtidm-safe-rl-group15/external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42/
```

The trained predictor can then be combined with the public CPO policy checkpoint.
Predictive evaluation deliberately requires an explicit upper pruning threshold
below 180 mg/dL because the paper does not publish the selected value.  Do not
reuse the legacy 250 mg/dL default.

For example, after selecting `170` and penalty magnitude `10` on a validation
split, run one clean paired condition with:

```bash
sbatch jobs/evaluate_shield.slurm \
  10 none 0 adolescent#002 170 10 22
```

Arguments are, in order: episodes, perturbation, policy seed, patient, calibrated
upper threshold, penalty magnitude, and evaluation seed.  The job evaluates
both no-shield and predictive-shield conditions with identical episode seeds.
Values `170` and `10` above are examples, not claimed paper parameters.  The
paper-aligned penalty grid is `5`, `10`, and `15`; the upper threshold and penalty
must be selected on validation patients before held-out testing.

## Login-node rules

Appropriate on the login node:

- `cd`, `ls`, `mkdir`, and small metadata checks;
- `scp`, `sftp`, or `rsync` file transfer;
- `accinfo`, `budget-overview`, `myquota`, `sinfo`, and `scontrol`;
- `sbatch`, `squeue`, `sacct`, `scancel`, and inspecting modest log files.

Do not run on the login node:

- predictor training or evaluation;
- `pip install` or environment setup;
- Python tests, benchmarks, or smoke tests;
- GPU checks or simulations;
- CPU-, memory-, or I/O-intensive preprocessing.

## Failure handling

- `Invalid account or account/partition combination`: stop and ask the course
  staff which account or reservation must be supplied. Do not switch to a public
  GPU partition without confirming budget authorization.
- Setup log cannot download packages or data: keep the failed log and ask the
  course staff/SURF whether outbound access is available from `staging`; do not
  move the download workload to the login node.
- `CUDA is unavailable`: the GPU job exits before training. Inspect the partition
  and module log instead of accepting CPU fallback.
- Transition count is not 180: rerun the setup Slurm job and inspect its log.
- Job exceeds wall time: use the smoke-test timing and resubmit with a justified
  longer limit. The current trainer does not provide robust mid-epoch resume.
- `NODE_FAIL`: retain the log and inspect accounting; SURF documents that genuine
  node failures are reimbursed automatically.

## SURF references

- Login-node policy and connection:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660216/Connecting+to+the+system>
- SSH host fingerprints:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/92668106/Snellius+SSH+hostkey+fingerprints>
- File transfer:
  <https://servicedesk.surf.nl/wiki/pages/viewpage.action?navigatingVersions=true&pageId=92668164>
- Account and budget information:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/300974118/Getting+account+and+budget+information>
- Job scripts:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660220/Writing+a+job+script>
- Example GPU job scripts:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660234/Example+job+scripts>
- Queue monitoring:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660228/Interacting+with+the+job+queue>
- Partitions:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions>
- Filesystems and quotas:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems>
- Current PyTorch-on-2025-stack guidance:
  <https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30661845/Snellius+known+issues>
