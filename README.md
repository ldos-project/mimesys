<img src="mimesys.png" alt="Mimesys" width="120">

# Mimesys: Generating Realistic Executable Testing Environments from Resource Usage Traces

Mimesys generates executable workloads that reproduce the hardware performance traces of target applications using a conditional diffusion model.

Diffusion-based system emulation framework. Trains a conditional diffusion model to generate executable workloads that reproduce hardware performance traces of target applications.


---
## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [1. Target Machine Setup](#1-target-machine-setup)
- [2. Data Collection](#2-data-collection)
- [3. Training](#3-training)
- [4. RL Fine-tuning](#4-rl-fine-tuning)
- [5. Inference Server & Client](#5-inference-server--client)

---

## Requirements

- Python 3.10
- CUDA 12.x
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

---

## Installation

```bash
uv sync
```

This creates a `.venv` in the repo root, installs all pinned dependencies from `uv.lock`, and installs the `mimesys` package in editable mode. To run any command inside the environment, prefix it with `uv run`.

To activate the venv directly (optional):

```bash
source .venv/bin/activate
```

---

## 1. Target Machine Setup

Data collection requires a cluster of machines accessible over SSH. Each node runs the benchmark and ships results back to the controller. The scripts in `worker_scripts/` automate this setup.

### SSH configuration

Fill in `worker_scripts/config.py` with your credentials, paths, and the list of worker hostnames:

```python
USERNAME         = "your_username"
PRIVATE_KEY_PATH = "~/.ssh/id_rsa"
LOCAL_HOME_DIR   = "/home/your_username"       # home dir on the controller
REMOTE_HOME_DIR  = "/users/your_username"      # home dir on the workers
MY_HOSTNAME      = "controller.example.com"    # must be resolvable from the workers
HOSTNAMES        = ["worker-01.example.com", "worker-02.example.com", ...]
```

`MY_HOSTNAME` is the address workers use to `scp` results back to the controller, so it cannot be `localhost`.

### Installing dependencies on each worker

Run the following command to install all required dependencies for executing synthesized workloads and collecting profiling metrics.

**Command**

```bash
cd worker_scripts
uv run python install_remote_dependencies.py
```

**How collection works on each machine**

During each active-learning round, the controller:

1. Writes execution plans (HDF5 files) to per-node directories and transfers them over SSH.
2. Each node runs `worker_scripts/collect_mimesys_metrics.sh`, which invokes the benchmark with the assigned plans and collects hardware performance counters.
3. Results are zipped, copied back to the controller via `scp`, then parsed and filtered before being added to the training dataset.

---

## 2. Data Collection

Training data is collected via active learning (collection/collect_training_data.py). Each round proposes candidate stressor compositions, predicts their hardware metrics with a Random-Forest surrogate, selects the candidates that fill gaps in the observed metric space, and profiles them on the worker machines.

**Command**

```bash
cd mimesys/collection
N_ROUNDS_NEW=20 uv run python collect_training_data.py
```

Configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OUT_DIR` | `~/mimesys_training_data/al_singleRF_hullfill_v1` | Output directory (`round_N/chunk_*/`) |
| `SEED_FROM` | `~/mimesys_training_data/training_data_round1_1sec` | Existing collection providing `metrics_range_dict.pkl` for metric normalization (produced by the dataloader cache of a prior collection) |
| `N_ROUNDS_NEW` | `3` | Active-learning rounds after round 0 |
| `N_CANDIDATES` | `512` | Candidate pool size per round |
| `N_SELECT` | `128` | Plans selected and profiled per round |

Completed rounds are detected on restart and skipped, so re-running the script resumes an interrupted collection.

**Overview**

Round 0 — Initialization

* A one-hot sweep (initial_candidates) covers each of the stressors in isolation across varying thread counts and weight scales. This anchors the dataset with ground-truth single-stressor behavior at the extremes of the metric space.

Rounds 1+ — Active Learning

1. Candidate generation — alternates by round parity: odd rounds sample random stressor compositions (sample_plans); even rounds synthesize thread-disjoint unions of already-observed reference pairs, targeting empty cells of the metric grid.

2. Surrogate scoring — a bank of single-output Random-Forest regressors (one per metric) is trained on all collected (action → metric) pairs and predicts each candidate's metric vector.

3. Hull-gap selection — the observed metric space is gridded across selected metric-pair subspaces; candidates whose predicted metrics land in empty or sparsely occupied cells are chosen first (multi-pair by default; a K-stratified variant that balances the batch across active-thread counts is available via `K_STRATIFY=1`), with farthest-point-sampling novelty as a fallback.

4. Profiling — selected plans are written as per-worker HDF5 chunks, distributed to the hosts in `worker_scripts/config.py`, and profiled; results are pulled back to the controller as they complete.

---

## 3. Training

All training commands are run from `mimesys/training/`.

**Supervised pretraining**

```bash
cd mimesys/training
CUDA_VISIBLE_DEVICES=0 uv run python trainer.py +exps=pretrain
```

The denoiser architecture is selected by `model_arch` — `dit` (default, a 2D DiT-style transformer where each stressor×thread cell is one token, with AdaLN-zero conditioning), `unet`, or `mlp`:

```bash
uv run python trainer.py +exps=pretrain model_arch=unet
```

Checkpoints only load into the architecture they were trained with, so pass the matching `model_arch` when resuming or serving older U-Net checkpoints.

**Multi-GPU**

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python trainer.py +exps=pretrain \
    train.trainer.devices=2 \
    train.trainer.strategy=ddp
```

**Resume from checkpoint**

```bash
uv run python trainer.py +exps=pretrain \
    train.trainer.ckpt_path=/path/to/diffusion-epoch=999.ckpt
```

**Training config example**

```yaml
# mimesys/conf/exps/pretrain.yaml  (train + model sections)
train:
  trainer:
    trainer_model_name: MimesysTrainer
    max_epochs: 2000
    devices: 1
    check_val_every_n_epoch: 500
  optim:
    lr: 1e-4
  callbacks:
    checkpoint:
      dirpath: diffusion/mimesys_pretrain_v1
      every_n_epochs: 500
      monitor: epoch
  run_train: true
  run_test: false
  use_rl: false
  prev_state_lambda: 0.0

model:
  unet:
    input_dim: 20           # thread dimension of the action grid
  context:
    input_dim: 23           # trace feature dimension
    action_dim: 33
    num_heads: 4
    num_layers: 6
    hidden_dim: 256
    dropout: 0.1
  diffusion:
    n_timesteps: 25
    cfg_args:
      cfg_drop_prob: 0.1
      cfg_guide_w: 3

log:
  project_name: mimesys_pretrain
  run_name: diffusion/mimesys_pretrain_v1
```

Training metrics are logged to [Weights & Biases](https://wandb.ai) under `log.project_name` / `log.run_name`.

---

## 4. RL Fine-tuning

RL fine-tuning uses DDPO with a profiling-based reward. Start from a pretrained supervised checkpoint.

**Command**

```bash
cd mimesys/training
CUDA_VISIBLE_DEVICES=0 uv run python trainer.py +exps=rl_finetuning
```

**RL config example**

```yaml
# mimesys/conf/exps/rl_finetuning.yaml  (train section)
train:
  trainer:
    trainer_model_name: MimesysTrainer
    max_epochs: 10000
    devices: 1
    check_val_every_n_epoch: 10
    ckpt_path: <FILL_IN_CKPT_PATH>  # start from supervised ckpt
  optim:
    lr: 3e-6                        # keep low to avoid catastrophic forgetting
  use_rl: true
  prev_state_lambda: 0.0
  ddpo:
    num_inner_epochs: 1
    num_batches_per_episode: 14
    reward_type: profiling          # live benchmark reward via SSH
    io_reward_weight: 1.0
    kl_coef: 0.05                   # KL penalty to pretrained distribution
  callbacks:
    checkpoint:
      dirpath: diffusion/mimesys_finetune_rl_v1
      every_n_epochs: 10
      monitor: epoch
      save_last: true
  async_validation: false

log:
  project_name: mimesys_rl
  run_name: diffusion/mimesys_finetune_rl_v1
```

If the supervised checkpoint was trained with a different architecture than the current `model_arch` default (`dit`), pass the matching one, e.g. `model_arch=unet`.

The `profiler` section must be set (same SSH credentials as data collection). Remote machines run the benchmark and return the reward signal each episode.

---

## 5. Inference Server & Client

### Server

```bash
uv run python -m mimesys.inference \
    --ckpt /path/to/diffusion-epoch=999.ckpt

# Custom port and experiment config
uv run python -m mimesys.inference \
    --ckpt /path/to/last.ckpt \
    --exp pretrain \
    --port 8000

# With remote profiling endpoint enabled
uv run python -m mimesys.inference \
    --ckpt /path/to/last.ckpt \
    --enable_profiling

# Choose devices
uv run python -m mimesys.inference \
    --ckpt /path/to/last.ckpt \
    --device cpu # or cuda
```

| Flag | Default | Description |
|---|---|---|
| `--ckpt` | required | Path to `.ckpt` checkpoint |
| `--exp` | `pretrain` | Hydra experiment config name |
| `--port` | `8000` | HTTP port |
| `--host` | `0.0.0.0` | Bind address |
| `--enable_profiling` | off | Enable `POST /profile` (requires SSH access to the profiling workers) |
| `--device` | auto | Force `cuda` or `cpu` |


### Client

Generate an execution plan (HDF5) from a time-series resource usage trace file. The trace file uses the [HPCPerfStats](https://github.com/TACC/HPCPerfStats) format.

```bash
uv run python -m mimesys.inference.client generate-from-file \
    --file /path/to/stats-workload.txt \
    --method diffusion \
    --output execution_plan_series.h5
```

Using the generated `h5` file, you can run a synthetic workload on a target machine:

```bash
# From the `fleetbench` directory on the target machine:

# Copy the h5 file to the machine first, then place it in the execution plans directory
cp execution_plan_series.h5 fleetbench/mimesys/execution_plans/

# Run the synthetic workload (HOME_PATH = directory containing fleetbench/ and HPCPerfStats/)
HOME_PATH=$HOME
MIMESYS_ITERS=1 MIMESYS_SLEEP=0 ACTION_PROFILING_CACHE_DIR=${HOME_PATH}/fleetbench ACTION_LIST_PATH=${HOME_PATH}/fleetbench/fleetbench/mimesys/mimesys_actions.txt TACC_STATS_DIR=${HOME_PATH}/HPCPerfStats/monitor/src sudo bazel run --config=clang --config=opt fleetbench/mimesys:mimesys_benchmark -- --benchmark_filter="BM_Mimesys"
```

To generate a workload and profile it in a single command:

```bash
uv run python -m mimesys.inference.client profile-from-file \
    --file /path/to/stats-workload.txt \
    --method diffusion \
    --output metrics.png
```

