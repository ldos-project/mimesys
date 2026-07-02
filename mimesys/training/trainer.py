import asyncio
import json
import os
import threading
import warnings
from collections import defaultdict, deque
from copy import deepcopy

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.distributions import Normal

from mimesys.preprocessing.dataloader import CustomDataLoader
from mimesys.preprocessing.system_trace import unnormalize_trace
from mimesys.schema.constants import max_time_steps, num_metrics_to_collect
from mimesys.training.utils import (
    EMA,
    calculate_fleetbench_action_error,
    initialize_callbacks,
    initialize_diffusion_model,
    initialize_logger,
    load_clip_model,
    load_transformer_model,
    write_fleetbench_actions_to_h5_file,
)
from mimesys.collection.profiling_server import (
    InitializeRequest,
    ProfileRequest,
    Profiler,
)
from mimesys.training.replay_buffer import ReplayBuffer

warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# Background event loop used for fire-and-forget async profiling calls.
loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _compute_predictions(model, batch):
    """Run one reverse-diffusion pass and return (predicted_actions, mse, log_probs)."""
    x0 = batch["label"]
    trace = batch["clean_trace"]
    prev_states = batch["prev_label"]
    chains, log_probs = model.p_sample_loop_with_logprobs(
        x0.shape, {"metric": trace, "prev_action": prev_states}
    )
    predicted_labels = chains[-1].cpu().detach().numpy()
    mse = float(np.mean((x0.cpu().detach().numpy() - predicted_labels) ** 2))
    return predicted_labels, mse, log_probs


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MimesysTrainer(pl.LightningModule):
    def __init__(self, diffusion_model, cfg):
        super().__init__()
        self.cfg = cfg
        self.trainer_model = diffusion_model
        self.ema = EMA(cfg.train.optim.ema.beta, cfg.train.optim.ema.step_start_ema)
        self.ema_model = deepcopy(self.trainer_model).eval().requires_grad_(False)
        self.test_machines = cfg.data.test_machines
        self.num_epochs_trained = 0
        self.metric_list = sorted(list(cfg.data.trace_range.keys()))

        self.profiler = Profiler(InitializeRequest(
            user_name=cfg.profiler.user_name,
            private_key_path=cfg.profiler.private_key_path,
            worker_host_names=self.test_machines,
            my_hostname=cfg.profiler.my_hostname,
        ))
        self.clip_model = None
        self.trace_predictor_model = None

        self.replay_buffer = ReplayBuffer(cfg.data.batch_size, device="cuda")
        self.trainer_model_old = None
        self.min_count = 16
        self.epsilon_clip = 0.3
        self.advantage_deque = deque(maxlen=32 * 64)
        self.samples = []
        self.ref_model = None  # frozen pretrained snapshot for KL regularization

        # Dynamic reward-weight state (Lagrangian-style adaptive λ_m). Enabled via
        # train.ddpo.dynamic_weights=true. After each profiling step the trainer
        # reads the RAW per-metric medians, appends to a rolling window, and updates
        # λ_m toward the relative violation from per-metric targets (pretrain K=200
        # baselines by default). λ_m is clamped to [0.1, dynamic_max_weight] and
        # passed into the NEXT ProfileRequest as the static io/llc/membw/cpu_reward_weight.
        ddpo_cfg = self.cfg.train.ddpo
        self.dyn_enabled = bool(ddpo_cfg.get("dynamic_weights", False))
        if self.dyn_enabled:
            self.dyn_window = int(ddpo_cfg.get("dynamic_window", 5))
            self.dyn_step   = float(ddpo_cfg.get("dynamic_step", 0.5))
            self.dyn_max    = float(ddpo_cfg.get("dynamic_max_weight", 5.0))
            self.dyn_lambdas = {
                "io":               float(ddpo_cfg.get("io_reward_weight", 1.0)),
                "l3_cache_usage":   float(ddpo_cfg.get("llc_reward_weight", 1.0)),
                "memory_bandwidth": float(ddpo_cfg.get("membw_reward_weight", 1.0)),
                "cpu":              float(ddpo_cfg.get("cpu_reward_weight", 1.0)),
            }
            # Targets default to pretrain v10 concat K=200 medians
            self.dyn_targets = {
                "io":               float(ddpo_cfg.get("dynamic_target_io",  0.0232)),
                "l3_cache_usage":   float(ddpo_cfg.get("dynamic_target_llc", 0.0362)),
                "memory_bandwidth": float(ddpo_cfg.get("dynamic_target_bw",  0.0330)),
                "cpu":              float(ddpo_cfg.get("dynamic_target_cpu", 0.0367)),
            }
            self.dyn_history = {k: deque(maxlen=self.dyn_window) for k in self.dyn_lambdas}
            print(f"[dyn-λ] enabled; window={self.dyn_window} step={self.dyn_step} "
                  f"max={self.dyn_max} targets={self.dyn_targets} init λ={self.dyn_lambdas}")

    # ------------------------------------------------------------------
    # Auxiliary model loading
    # ------------------------------------------------------------------

    def load_clip_model(self, cfg):
        self.clip_model = load_clip_model(cfg, device="cuda")

    def load_trace_predictor_model(self, cfg):
        self.trace_predictor_model = load_transformer_model(cfg)

    # ------------------------------------------------------------------
    # Artifact I/O
    # ------------------------------------------------------------------

    def write_system_traces_to_file(self, dirname: str, trace_list) -> dict:
        trace_range = self.cfg.data.trace_range
        metrics_dict = {}
        for metric_idx, metric in enumerate(self.metric_list):
            lo, hi = trace_range[metric][0], trace_range[metric][1]
            start = metric_idx * num_metrics_to_collect
            end = (metric_idx + 1) * num_metrics_to_collect
            metrics_dict[metric] = unnormalize_trace(
                [trace_list.tolist()[start:end]], lo, hi
            )[0]
        with open(os.path.join(dirname, "system_traces.json"), "w") as f:
            json.dump(metrics_dict, f)
        return metrics_dict

    def write_labels_to_file(self, dirname: str, action_list, file_name: str = "label"):
        write_fleetbench_actions_to_h5_file(
            action_list, os.path.join(dirname, f"{file_name}.h5")
        )
        return action_list

    # ------------------------------------------------------------------
    # Reward computation (DDPO outer loop)
    # ------------------------------------------------------------------

    def sample_and_calculate_rewards(self, batch, exp_path="training"):
        N = batch["label"].shape[0]
        T = self.trainer_model.n_timesteps
        x0 = batch["label"]
        trace = batch["clean_trace"]
        prev_trace = batch["prev_clean_trace"]
        prev_labels = batch["prev_label"]
        is_training_data = batch["training_data"]

        prev_states = -torch.ones_like(x0).cuda()
        # When skip_prev_state=True (metrics_only encoder convention): no
        # prev_action signal is meaningful, so we leave prev_states at -1 (idle
        # in scaled space), skip the prev_medoid diffusion, and write 1-window
        # plans below. The deploy then has cycle (curr, sleep) × ITERS.
        skip_prev_state = bool(self.cfg.train.ddpo.get("skip_prev_state", False))
        if skip_prev_state:
            pass  # leave prev_states at -ones
        elif (prev_medoid_n := int(self.cfg.train.ddpo.get("prev_medoid_n", 1))) > 1:
            # Match predict_autoregressive_h5.py's prev_raw computation for one step:
            #   prev_raw_init = zeros (raw [0,1])  →  scaled = -ones (idle)
            #   for K candidates: diffusion(cond={"metric": prev_trace, "prev_action": idle})
            #   medoid = argmin_k sum_j mean_d |a_j[d] - a_k[d]|     (sum-L1, dim-averaged)
            # The selected medoid (scaled [-1,+1]) is used as prev_action input to the
            # subsequent curr-action diffusion call (matches inference t=0 → t=1 transition).
            K = prev_medoid_n
            prev_trace_K  = prev_trace.repeat_interleave(K, dim=0).cuda()     # (N*K, dim)
            prev_states_K = prev_states.repeat_interleave(K, dim=0)            # (N*K, S, T)  all -1 (idle)
            shape_K       = (N * K,) + tuple(x0.shape[1:])
            with torch.no_grad():
                chains_K, _ = self.trainer_model.p_sample_loop_with_logprobs(
                    shape_K, {"metric": prev_trace_K, "prev_action": prev_states_K}
                )
            final = chains_K[-1].view(N, K, *x0.shape[1:])                     # (N, K, S, T) scaled
            # Medoid by min sum-L1 (dim-averaged), GPU-resident; identical formula to inference.
            flat = final.reshape(N, K, -1)                                     # (N, K, S*T)
            # pairwise: (N, K_query, K_other, D) → mean over D → sum over K_other → (N, K_query)
            d = (flat.unsqueeze(2) - flat.unsqueeze(1)).abs().mean(dim=-1).sum(dim=2)  # (N, K)
            medoid_idx = d.argmin(dim=1)                                       # (N,)
            prev_states = final[torch.arange(N, device=final.device), medoid_idx]   # (N, S, T)
        else:
            chains, _ = self.trainer_model.p_sample_loop_with_logprobs(
                x0.shape, {"metric": prev_trace, "prev_action": prev_states}
            )
            prev_states = chains[-1]
        if not skip_prev_state:
            for idx in range(N):
                if is_training_data[idx]:
                    prev_states[idx] = prev_labels[idx]

        trace = (trace + torch.randn_like(trace) * 1e-4).cuda()

        # === curr_action selection ===
        # curr_medoid_n=K: sample K candidates per row in one inflated-batch diffusion
        # call, pick the per-row medoid (min sum-L1 to other candidates) — matches
        # predict_autoregressive_h5.py's curr-action selection. The medoid's chain
        # and log_probs are kept for the DDPO gradient (other K-1 chains discarded).
        # Falls back to best_of_n=1 single-sample when curr_medoid_n<=1.
        curr_medoid_n = int(self.cfg.train.ddpo.get("curr_medoid_n", 1))
        if curr_medoid_n > 1:
            K = curr_medoid_n
            trace_K       = trace.repeat_interleave(K, dim=0).cuda()         # (N*K, dim)
            prev_states_K = prev_states.repeat_interleave(K, dim=0).cuda()    # (N*K, S, T)
            shape_K       = (N * K,) + tuple(x0.shape[1:])
            with torch.no_grad():
                chains_K, log_probs_K = self.trainer_model.p_sample_loop_with_logprobs(
                    shape_K, {"metric": trace_K, "prev_action": prev_states_K}
                )
            # chains_K: (T+1, N*K, S, T_a) ; log_probs_K: (T, N*K)
            T_p1 = chains_K.shape[0]
            T_dif = log_probs_K.shape[0]
            chains_NK    = chains_K.view(T_p1, N, K, *x0.shape[1:])
            log_probs_NK = log_probs_K.view(T_dif, N, K)
            final = chains_NK[-1]                                              # (N, K, S, T_a)
            flat  = final.reshape(N, K, -1)                                    # (N, K, S*T)
            d = (flat.unsqueeze(2) - flat.unsqueeze(1)).abs().mean(dim=-1).sum(dim=2)  # (N, K)
            medoid_idx = d.argmin(dim=1)                                       # (N,)
            row_idx = torch.arange(N, device=final.device)
            chains    = chains_NK[:, row_idx, medoid_idx]                      # (T+1, N, S, T_a)
            log_probs = log_probs_NK[:, row_idx, medoid_idx]                   # (T, N)
            all_chains    = [chains]
            all_log_probs = [log_probs]
            best_of_n     = 1                                                  # downstream loops over best_of_n
        else:
            best_of_n = int(self.cfg.train.ddpo.get("best_of_n", 1))
            all_chains = []
            all_log_probs = []
            with torch.no_grad():
                for _k in range(best_of_n):
                    chains_k, log_probs_k = self.trainer_model.p_sample_loop_with_logprobs(
                        x0.shape, {"metric": trace, "prev_action": prev_states}
                    )
                    all_chains.append(chains_k)
                    all_log_probs.append(log_probs_k)
            # Default-path tensors (matching the K=1 case) reference the first sample;
            # we overwrite chains/log_probs after profiling selects per-sample winners.
            chains = all_chains[0]
            log_probs = all_log_probs[0]
        predicted_labels = chains[-1].cpu().detach().numpy()
        predicted_actions_tensor = (
            torch.tensor(predicted_labels).float().cuda().view(N, -1)
        )

        reward_type = self.cfg.train.ddpo.reward_type
        if reward_type == "transformer":
            mse = self.trace_predictor_model.train_step(predicted_actions_tensor, trace)
            trace.detach()
            del trace
            rewards = 10 / (1 + mse)

        elif reward_type == "profiling":
            checkpoint_dir = self.cfg.train.callbacks.checkpoint.dirpath
            # Write N * best_of_n batch dirs: dir index = sample_idx * best_of_n + k
            for sample_idx, prev_state in enumerate(prev_states):
                for k in range(best_of_n):
                    action_k = all_chains[k][-1][sample_idx].cpu().detach().numpy()
                    dir_idx = sample_idx * best_of_n + k
                    dirname = os.path.join(
                        checkpoint_dir, exp_path,
                        f"step_{self.global_step}_batch_{dir_idx}",
                    )
                    os.makedirs(dirname, exist_ok=True)
                    if k == 0:
                        # target metric is shared across k; write only once equivalently
                        self.write_system_traces_to_file(dirname, batch["clean_trace"][sample_idx])
                    else:
                        self.write_system_traces_to_file(dirname, batch["clean_trace"][sample_idx])
                    # Skip prev window entirely when skip_prev_state — the
                    # benchmark sees a 1-window plan; deploy cycle becomes
                    # (curr, sleep) × ITERS instead of (prev, curr, sleep) × ITERS.
                    plan_windows = (
                        [action_k] if skip_prev_state
                        else [prev_state, action_k]
                    )
                    write_fleetbench_actions_to_h5_file(
                        plan_windows,
                        os.path.join(dirname, "predicted_actions_trainer_0.h5"),
                    )

            # Current step's reward weights: dynamic λ if enabled, else static config.
            if self.dyn_enabled:
                w_io  = float(self.dyn_lambdas["io"])
                w_llc = float(self.dyn_lambdas["l3_cache_usage"])
                w_bw  = float(self.dyn_lambdas["memory_bandwidth"])
                w_cpu = float(self.dyn_lambdas["cpu"])
            else:
                w_io  = self.cfg.train.ddpo.get("io_reward_weight", 1.0)
                w_llc = self.cfg.train.ddpo.get("llc_reward_weight", 1.0)
                w_bw  = self.cfg.train.ddpo.get("membw_reward_weight", 1.0)
                w_cpu = self.cfg.train.ddpo.get("cpu_reward_weight", 1.0)
            # Reward-slice indexing: skip_prev_state produces a (curr, sleep)×ITERS
            # measurement stream, so pick even indices (period=2, offset=0). The
            # historical 2-window deploy uses (period=3, offset=1) to land on the
            # curr entry of each (prev, curr, sleep) triplet.
            default_slice_period = 2 if skip_prev_state else 3
            default_slice_offset = 0 if skip_prev_state else 1
            profile_request = ProfileRequest(
                validation_data_path=f"{checkpoint_dir}/{exp_path}",
                my_destination_path=self.cfg.profiler.destination_path,
                step=self.global_step,
                num_batches=N * best_of_n,
                num_trials=1,
                logger=self.trainer.logger,
                model_type="trainer",
                io_reward_weight=w_io,
                low_resource_penalty_weight=self.cfg.train.ddpo.get("low_resource_penalty_weight", 0.0),
                membw_reward_weight=w_bw,
                llc_reward_weight=w_llc,
                cpu_reward_weight=w_cpu,
                reward_slice_period=int(self.cfg.train.ddpo.get("reward_slice_period", default_slice_period)),
                reward_slice_offset=int(self.cfg.train.ddpo.get("reward_slice_offset", default_slice_offset)),
                use_raw_mean_reward=bool(self.cfg.train.ddpo.get("use_raw_mean_reward", False)),
            )
            print("Profile request:", profile_request)
            avg_emd_by_metrics, avg_emd_by_batch, metrics_std_by_batch = asyncio.run(
                self.profiler.profile(
                    profile_request, self.cfg.data.trace_range,
                    period=2, duration=6, aggregate_time_series=True,
                )
            )

            # Log per-metric RAW L1 errors (unweighted by λ) to wandb so we can see
            # whether mem-BW / LLC actually improved, independent of the reward signal.
            # Aggregates avg_cpu_utilizations* keys into a single "cpu".
            if avg_emd_by_metrics:
                _cpu_vals = []
                _all_raw_means = []
                for m_name, errs in avg_emd_by_metrics.items():
                    if not errs:
                        continue
                    mean_err = float(np.mean(errs))
                    _all_raw_means.append(mean_err)
                    # Legacy aggregate buckets — still log under the old short names
                    # for backwards-compatible dashboards.
                    if m_name == "io":
                        self.log("emd/io",    mean_err, prog_bar=False, logger=True, on_epoch=True)
                    elif m_name == "l3_cache_usage":
                        self.log("emd/llc",   mean_err, prog_bar=False, logger=True, on_epoch=True)
                    elif m_name == "memory_bandwidth":
                        self.log("emd/membw", mean_err, prog_bar=False, logger=True, on_epoch=True)
                    elif m_name.startswith("avg_cpu_utilizations"):
                        _cpu_vals.append(mean_err)
                    # ALSO log every metric under its own name so v2 keys
                    # (io_read, io_write, mb_read, mb_write, pqos_*) get
                    # individual wandb charts.
                    self.log(f"emd/m_{m_name}", mean_err,
                             prog_bar=False, logger=True, on_epoch=True)
                if _cpu_vals:
                    self.log("emd/cpu", float(np.mean(_cpu_vals)),
                             prog_bar=False, logger=True, on_epoch=True)
                # Mean RAW L1 across all metric keys (what the reward WOULD be if all λ=1).
                if _all_raw_means:
                    self.log("emd/raw_mean", float(np.mean(_all_raw_means)),
                             prog_bar=False, logger=True, on_epoch=True)
            # WEIGHTED L1 (the actual value that becomes −reward, before std penalty).
            # avg_emd_by_batch is the per-batch mean of λ_m · err_m — exactly what's
            # turned into rewards_flat = −emd_error_flat − 0.05·metrics_std_flat below.
            if len(avg_emd_by_batch) > 0:
                self.log("emd/weighted_mean", float(np.mean(avg_emd_by_batch)),
                         prog_bar=True, logger=True, on_epoch=True)

            # Update dynamic λ_m via SOFTMAX over per-family rolling violations.
            # Properties (vs the old Lagrangian PI update):
            #   * sum(λ_m) is held constant at K=4 → total reward magnitude
            #     does not inflate over time (fixes the "reward gets worse even when
            #     emd improves" pathology caused by unbounded λ growth).
            #   * If a metric is satisfied (violation==0) it doesn't get extra weight,
            #     but it also doesn't strangle the others — softmax handles it gracefully.
            #   * Targets being "unreachable" doesn't matter — softmax cares about
            #     the RELATIVE ordering of violations across families.
            # dyn_step is reinterpreted as softmax temperature τ; larger τ = more peaked.
            if self.dyn_enabled and avg_emd_by_metrics:
                # 1) per-family rolling median (aggregate cpu sub-keys to one family)
                for m_name, errs in avg_emd_by_metrics.items():
                    if not errs:
                        continue
                    key = "cpu" if m_name.startswith("avg_cpu_utilizations") else m_name
                    if key in self.dyn_lambdas:
                        # avg_emd_by_metrics values are RAW (unweighted) — see profiling_server.py
                        self.dyn_history[key].append(float(np.median(errs)))
                # 2) softmax(τ · max(0, rolling/target − 1)) → λ_m
                order = ["io", "l3_cache_usage", "memory_bandwidth", "cpu"]
                rolling_arr, viol_arr = [], []
                for k in order:
                    if self.dyn_history[k]:
                        r = float(np.median(list(self.dyn_history[k])))
                    else:
                        r = self.dyn_targets[k]   # no data yet → uniform
                    rolling_arr.append(r)
                    viol_arr.append(max(0.0, r / max(self.dyn_targets[k], 1e-9) - 1.0))
                viol_arr = np.asarray(viol_arr, dtype=np.float64)
                tau = float(self.dyn_step)
                # numerically-stable softmax
                z = tau * viol_arr; z -= z.max()
                w = np.exp(z); w = w / w.sum()
                K = float(len(order))
                lambdas = K * w   # sums to K; uniform λ=1 each when all violations equal
                for i, k in enumerate(order):
                    self.dyn_lambdas[k] = float(lambdas[i])
                update_info = {k: (round(rolling_arr[i], 4),
                                   round(viol_arr[i], 3),
                                   round(lambdas[i], 3))
                               for i, k in enumerate(order)}
                print(f"[dyn-w] (rolling, violation, λ): {update_info}")
                # Log to wandb
                _lam_key = {"io": "lambda/io", "l3_cache_usage": "lambda/llc",
                            "memory_bandwidth": "lambda/membw", "cpu": "lambda/cpu"}
                _viol_key = {"io": "violation/io", "l3_cache_usage": "violation/llc",
                             "memory_bandwidth": "violation/membw", "cpu": "violation/cpu"}
                for i, k in enumerate(order):
                    self.log(_lam_key[k],  float(lambdas[i]),
                             prog_bar=False, logger=True, on_epoch=True)
                    self.log(_viol_key[k], float(viol_arr[i]),
                             prog_bar=False, logger=True, on_epoch=True)

            emd_error_flat = torch.tensor(avg_emd_by_batch, dtype=torch.float32)
            metrics_std_flat = torch.tensor(metrics_std_by_batch, dtype=torch.float32)
            rewards_flat = -emd_error_flat - metrics_std_flat * 0.05

            # NaN-defense: a single NaN reward poisons mean/std in compute_advantage,
            # making *all* future advantages NaN forever. NaN appears when the
            # profiler's _parse_local_stats divides by len(normalized_metrics)=0
            # (plan crashed mid-profile, no samples in the second-slot filter).
            # Replace NaN/Inf with a "bad-but-not-poisoning" reward of -2.0, and
            # clip extremes to keep advantage normalization stable.
            if torch.isnan(rewards_flat).any() or torch.isinf(rewards_flat).any():
                n_nan = int(torch.isnan(rewards_flat).sum() + torch.isinf(rewards_flat).sum())
                rewards_flat = torch.nan_to_num(rewards_flat, nan=-2.0, posinf=2.0, neginf=-2.0)
                print(f"[reward-nan-defense] replaced {n_nan} NaN/Inf rewards with -2.0")
            rewards_flat = torch.clamp(rewards_flat, -2.0, 2.0)

            if best_of_n > 1:
                # Pad / truncate rewards_flat to expected size (N*best_of_n).
                # Workers occasionally drop a profile (timeout/scp fail) which would
                # leave fewer entries than expected; pad missing entries with the
                # mean so the reshape works and the "missing" k is unlikely to win.
                expected = N * best_of_n
                if rewards_flat.numel() < expected:
                    pad_val = rewards_flat.mean().item() if rewards_flat.numel() > 0 else 0.0
                    pad = torch.full((expected - rewards_flat.numel(),), pad_val,
                                     dtype=rewards_flat.dtype)
                    rewards_flat = torch.cat([rewards_flat, pad])
                    print(f"[best_of_n] padded {pad.numel()} missing rewards with mean={pad_val:.4f}")
                elif rewards_flat.numel() > expected:
                    rewards_flat = rewards_flat[:expected]
                # Reshape (N*K,) -> (N, K), pick best k per sample
                rewards_mat = rewards_flat.view(N, best_of_n)
                best_k = rewards_mat.argmax(dim=1)
                rewards = rewards_mat.gather(1, best_k.unsqueeze(1)).squeeze(1)
                # Pick chains/log_probs for the winning k per sample
                chains_picked_list = []
                log_probs_picked_list = []
                for i in range(N):
                    ki = int(best_k[i].item())
                    chains_picked_list.append(all_chains[ki][:, i, :, :])    # (T_diff+1, S, T_threads)
                    log_probs_picked_list.append(all_log_probs[ki][:, i])     # (T_diff+1,)
                chains = torch.stack(chains_picked_list, dim=1)               # (T_diff+1, N, S, T_threads)
                log_probs = torch.stack(log_probs_picked_list, dim=1)         # (T_diff+1, N)
            else:
                rewards = rewards_flat

            chains = torch.flip(chains, dims=[0])
            log_probs = torch.flip(log_probs, dims=[0])
            chains_batched = chains.permute(1, 0, 2, 3)
            log_probs_batched = log_probs.permute(1, 0)
            rl_aug = int(self.cfg.train.ddpo.get("rl_aug_factor", 1))
            for i in range(N):
                trajectory = chains_batched[i]
                log_prob = log_probs_batched[i]
                context = batch["clean_trace"][i]
                prev_trace_i = batch["prev_clean_trace"][i]
                try:
                    reward = rewards[i]
                except IndexError:
                    continue
                for t in range(1, T):
                    self.replay_buffer.add_to_buffer(
                        state=(trajectory[t + 1], torch.tensor(t), context, prev_trace_i),
                        action=trajectory[t],
                        reward=reward,
                        log_probs=log_prob[t],
                        final_state=trajectory[0],
                    )

                # Augmentation (rl_aug-1 extra copies). Always intra-socket perm; if
                # socket_swap_aug=True, half of the copies also swap socket 0<->1.
                # Threads 0-9 belong to socket 0, 10-19 to socket 1. LLC and BW
                # metrics are now socket-aggregated (single fields each), so no
                # per-socket metric swap is needed under socket swap.
                socket_swap_aug = bool(self.cfg.train.ddpo.get("socket_swap_aug", False))
                for aug_idx in range(rl_aug - 1):
                    do_socket_swap = socket_swap_aug and (aug_idx % 2 == 1)
                    if do_socket_swap:
                        # Within each socket intra-perm, then place socket-1 threads at
                        # positions 0..9 and socket-0 threads at 10..19.
                        p_s1 = torch.randperm(10, device=trajectory.device) + 10
                        p_s0 = torch.randperm(10, device=trajectory.device)
                        thread_perm = torch.cat([p_s1, p_s0])
                    else:
                        p0 = torch.randperm(10, device=trajectory.device)
                        p1 = torch.randperm(10, device=trajectory.device) + 10
                        thread_perm = torch.cat([p0, p1])

                    # action / trajectory shape (T_diff+1, S=13, T_threads=20) -> permute last axis
                    new_trajectory = trajectory.index_select(-1, thread_perm)

                    # context / prev_trace flat shape (23,): positions 0-19 are CPU cores,
                    # [20]=io, [21]=l3_cache_usage, [22]=memory_bandwidth (all aggregated).
                    new_context = context.clone()
                    new_context[:20] = context[thread_perm]
                    new_prev_trace = prev_trace_i.clone()
                    new_prev_trace[:20] = prev_trace_i[thread_perm]

                    for t in range(1, T):
                        self.replay_buffer.add_to_buffer(
                            state=(new_trajectory[t + 1], torch.tensor(t),
                                   new_context, new_prev_trace),
                            action=new_trajectory[t],
                            reward=reward,
                            log_probs=log_prob[t],
                            final_state=new_trajectory[0],
                        )

        return rewards

    def compute_advantage(self):
        T = self.trainer_model.n_timesteps
        if not self.replay_buffer.rewards:
            # Whole profiling round returned no stats (e.g., chunk parse failures,
            # ssh hiccups, or every plan landed in an empty chunk). Skip this epoch's
            # advantage computation instead of crashing torch.stack([]).
            print("[compute_advantage] replay_buffer empty — skipping (no profile data for this step)")
            return
        rewards_dedup = torch.tensor(
            self.replay_buffer.rewards[::T], dtype=torch.float32
        )
        # Defense: if any pre-poisoned NaN slipped past the reward filter, scrub here.
        rewards_dedup = torch.nan_to_num(rewards_dedup, nan=-2.0, posinf=2.0, neginf=-2.0)
        self.advantage_deque.extend(rewards_dedup)

        if len(self.advantage_deque) < self.min_count:
            mean = rewards_dedup.mean()
            std = rewards_dedup.std() + 1e-6
        else:
            deque_arr = np.asarray(self.advantage_deque, dtype=np.float64)
            deque_arr = deque_arr[np.isfinite(deque_arr)]
            mean = torch.tensor(float(np.mean(deque_arr)) if deque_arr.size else 0.0).float()
            std = torch.tensor(float(np.std(deque_arr)) if deque_arr.size else 1.0).float() + 1e-6

        rewards = torch.stack(self.replay_buffer.rewards).float()
        rewards = torch.nan_to_num(rewards, nan=-2.0, posinf=2.0, neginf=-2.0)
        advantages = (rewards - mean) / std if len(rewards_dedup) > 1 else rewards - mean
        clipped = torch.clip(advantages, -3.0, 3.0)
        # Belt-and-suspenders: should be impossible to reach here with NaN, but
        # if std is somehow NaN, fall back to centered-only.
        if not torch.isfinite(clipped).all():
            clipped = torch.nan_to_num(clipped, nan=0.0, posinf=3.0, neginf=-3.0)
            print(f"[advantage-nan-defense] clipped NaN/Inf to 0")
        print("Advantage:", clipped)
        self.replay_buffer.set_advantages(clipped)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint):
        # ref_model is recreated from the current model in on_train_start;
        # strip its keys so strict state_dict loading doesn't fail when resuming.
        state_dict = checkpoint.get("state_dict", {})
        ref_keys = [k for k in list(state_dict.keys()) if k.startswith("ref_model.")]
        for k in ref_keys:
            del state_dict[k]
        if ref_keys:
            print(f"[on_load_checkpoint] Stripped {len(ref_keys)} ref_model keys from checkpoint")
        # Warm-start support: when fine-tuning a NEW architecture (e.g. residual_prev
        # adds prev_head + gate) from an old checkpoint that lacks those params,
        # backfill the missing keys from the freshly-initialised current model so the
        # strict load succeeds and the new params keep their (zero) init.
        cur = self.state_dict()
        added = [k for k in cur if k not in state_dict]
        for k in added:
            state_dict[k] = cur[k]
        if added:
            print(f"[on_load_checkpoint] Backfilled {len(added)} new-param keys "
                  f"(e.g. {added[:3]}) from current init — warm-start mode")
            # New architecture ⇒ the saved optimizer/scheduler param groups no longer
            # match. on_train_start resets the optimizer regardless, so drop them so
            # Lightning doesn't fail restoring a mismatched optimizer state.
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            print("[on_load_checkpoint] Cleared optimizer/scheduler states (warm-start)")

    def on_train_start(self):
        optimizer = self.optimizers()
        optimizer.state = defaultdict(dict)
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.cfg.train.optim.lr

        # Freeze a snapshot of the initial model for KL regularization
        kl_coef = self.cfg.train.ddpo.get("kl_coef", 0.0) if self.cfg.train.use_rl else 0.0
        if kl_coef > 0.0:
            self.ref_model = deepcopy(self.trainer_model).eval().requires_grad_(False)
            print(f"KL regularization enabled with kl_coef={kl_coef}")

    def training_step(self, batch, batch_idx):
        if self.cfg.train.use_rl:
            T = self.trainer_model.n_timesteps
            if len(self.samples) <= batch_idx * (T - 1):
                return None

            samples = self.samples[batch_idx * (T - 1) : (batch_idx + 1) * (T - 1)]
            merged_sample: dict = defaultdict(list)
            for sample in samples:
                for k, v in sample.items():
                    merged_sample[k].extend(v)

            sample_batch = self.replay_buffer.collate(merged_sample)
            xt, t, diffusion_context_cond, prev_trace = sample_batch["states"]
            actions = sample_batch["actions"]
            log_probs_old = sample_batch["log_probs"]
            advantages = sample_batch["advantages"]
            x0 = sample_batch["final_states"]

            prev_states = -torch.ones_like(x0).cuda()
            chains, _ = self.trainer_model.p_sample_loop_with_logprobs(
                x0.shape, {"metric": prev_trace, "prev_action": prev_states}
            )
            prev_states = chains[-1]

            mean, _, log_var = self.trainer_model.p_mean_variance(
                xt, t, {"metric": diffusion_context_cond, "prev_action": prev_states}
            )
            std = torch.clip(torch.exp(0.5 * log_var), min=1e-6)
            log_prob_new = Normal(mean, std).log_prob(actions)
            log_prob_new = log_prob_new.mean(dim=list(range(1, log_prob_new.ndim)))

            ratio = torch.exp(log_prob_new - log_probs_old)
            surr1 = -ratio * advantages
            surr2 = -torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            loss = torch.max(surr1, surr2).mean()

            # Optional KL regularization: penalize deviation from pretrained model
            kl_coef = self.cfg.train.ddpo.get("kl_coef", 0.0)
            if kl_coef > 0.0 and self.ref_model is not None:
                with torch.no_grad():
                    mean_ref, _, log_var_ref = self.ref_model.p_mean_variance(
                        xt, t, {"metric": diffusion_context_cond, "prev_action": prev_states}
                    )
                std_ref = torch.clip(torch.exp(0.5 * log_var_ref), min=1e-6)
                kl_div = torch.distributions.kl_divergence(
                    Normal(mean, std), Normal(mean_ref, std_ref)
                ).mean()
                loss = loss + kl_coef * kl_div
                self.log("kl_div", kl_div, prog_bar=True, logger=True, on_epoch=True)

            self.log("advantage", advantages.mean(), prog_bar=True, logger=True, on_epoch=True)
        else:
            self.num_epochs_trained += 1
            context_cond = batch["clean_trace"] + torch.randn_like(batch["clean_trace"]) * 1e-3
            x0 = batch["label"]
            loss = self.trainer_model.loss(
                x0, {"metric": context_cond, "prev_action": batch["prev_label"]}, None
            )

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    # ---------------------------------------------------------------------------
    # Windowed-DTW RL path
    # ---------------------------------------------------------------------------
    def _init_windowed_pool(self):
        """Build the K-means medoid → W-sec window pool once. Caches on self."""
        from mimesys.training.windowed_dtw_rl import deploy_windowed_plans  # ensure module loads
        K = int(self.cfg.train.ddpo.get("windowed_kmean_k", 3000))
        W = int(self.cfg.train.ddpo.get("window_size", 5))
        dm = self.trainer.datamodule
        print(f"[windowed-dtw] building K={K} W={W} pool ...")
        pool = dm._sample_rl_test_data_kmean_windowed(
            self.cfg.data.test_data_path, dm.trace_range, K=K, window=W,
        )
        self._windowed_pool = pool
        # METRIC_MAX for DTW normalization (cpu fixed to 100; rest from trace_range)
        self._windowed_metric_max = {
            "cpu": 100.0,
            "io":  float(dm.trace_range["io"][1]),
            "llc": float(dm.trace_range["l3_cache_usage"][1]),
            "bw":  float(dm.trace_range["memory_bandwidth"][1]),
        }
        # pqos families — only added when trace_range has them (new pqos-enabled traces).
        for k_meas, k_range in [("pqos_ipc", "pqos_ipc"),
                                 ("pqos_llc", "pqos_llc_kb"),
                                 ("pqos_misses", "pqos_misses")]:
            if k_range in dm.trace_range:
                self._windowed_metric_max[k_meas] = float(dm.trace_range[k_range][1])
        print(f"[windowed-dtw] pool size {len(pool)} ; METRIC_MAX={self._windowed_metric_max}")

    def _on_train_epoch_start_windowed(self) -> None:
        """Windowed-DTW RL epoch driver. Replaces the standard merged_batch path.

        Per epoch:
          1. Sample N = num_batches_per_episode windowed entries from the pool
          2. Auto-regressively generate W actions per sample (with K=prev_medoid_n
             medoid selection at each step)
          3. Deploy each sample's W-window plan on a cluster host (1 sec/window)
          4. Compute per-family normalized DTW vs GT 5-sec metric → reward (avg of 4)
          5. Push to replay_buffer for the standard DDPO training_step downstream
        """
        from pathlib import Path
        import numpy as _np
        from mimesys.training.windowed_dtw_rl import (
            rollout_windowed_with_medoid, deploy_windowed_plans, dtw_reward_per_sample,
        )

        if not hasattr(self, "_windowed_pool"):
            self._init_windowed_pool()

        self.replay_buffer.clear()

        N = int(self.cfg.train.ddpo.num_batches_per_episode)
        W = int(self.cfg.train.ddpo.get("window_size", 5))
        prev_K = int(self.cfg.train.ddpo.get("prev_medoid_n", 32))

        if len(self._windowed_pool) < N:
            print(f"[windowed-dtw] WARN: pool ({len(self._windowed_pool)}) < N ({N}); replay")
            indices = _np.random.choice(len(self._windowed_pool), N, replace=True)
        else:
            indices = _np.random.choice(len(self._windowed_pool), N, replace=False)
        sel = [self._windowed_pool[int(i)] for i in indices]

        metric_seq = torch.tensor([s["metric_seq"] for s in sel], dtype=torch.float32, device="cuda")   # (N, W, 23)
        prev_first = torch.tensor([s["prev_clean_trace"] for s in sel], dtype=torch.float32, device="cuda")  # (N, 23)
        raw_gt_4d  = _np.array([s["raw_metric_seq_4d"] for s in sel], dtype=_np.float64)               # (N, W, 4)
        # Per-core CPU GT (W, 20). Only used when per_core_cpu_reward=true and the pool
        # entries contain `raw_metric_seq_percore` (older cached pools may not).
        if sel and "raw_metric_seq_percore" in sel[0]:
            raw_gt_percore = _np.array([s["raw_metric_seq_percore"] for s in sel],
                                       dtype=_np.float64)                                              # (N, W, 20)
        else:
            raw_gt_percore = None
        # Pqos GT (W, 3) = [ipc, llc_kb, misses]. Only present if test traces had
        # pqos.log siblings; otherwise raw_metric_seq_pqos is None per-entry.
        if sel and sel[0].get("raw_metric_seq_pqos") is not None:
            raw_gt_pqos = _np.array([s["raw_metric_seq_pqos"] for s in sel],
                                     dtype=_np.float64)                                                # (N, W, 3)
        else:
            raw_gt_pqos = None

        # ---- (2) auto-regressive rollout
        chains_per_step, log_probs_per_step, plans_raw = rollout_windowed_with_medoid(
            self.trainer_model, metric_seq, prev_first, window=W, prev_medoid_n=prev_K
        )
        plans_raw_np = plans_raw.cpu().numpy()                                                          # (N, W, S, T_a)

        # ---- (3) deploy each N-window plan on the cluster
        out_dir = Path(self.cfg.train.callbacks.checkpoint.dirpath) / "training" / f"step_{self.global_step}_windowed"
        iters_per_window = int(self.cfg.train.ddpo.get("mimesys_iters_per_window", 1))
        # Use the data.test_machines list (from the hydra override) explicitly so the
        # deploy doesn't silently fall back to worker_scripts/config.HOSTNAMES, which
        # may point at a different cluster than what training intends.
        measured = deploy_windowed_plans(plans_raw_np, out_dir, window_size=W,
                                         hosts=list(self.test_machines),
                                         iters_per_window=iters_per_window)

        # ---- (4) DTW reward per sample
        # Pick up per-family weights and the optional per-core CPU term.
        use_percore = bool(self.cfg.train.ddpo.get("per_core_cpu_reward", False))
        reward_weights = {
            "cpu":          float(self.cfg.train.ddpo.get("cpu_reward_weight",   1.0)),
            "io":           float(self.cfg.train.ddpo.get("io_reward_weight",    1.0)),
            "llc":          float(self.cfg.train.ddpo.get("llc_reward_weight",   1.0)),
            "bw":           float(self.cfg.train.ddpo.get("membw_reward_weight", 1.0)),
            "per_core_cpu": float(self.cfg.train.ddpo.get("per_core_cpu_reward_weight", 1.0)),
            "pqos_ipc":     float(self.cfg.train.ddpo.get("pqos_ipc_reward_weight",    0.0)),
            "pqos_llc":     float(self.cfg.train.ddpo.get("pqos_llc_reward_weight",    0.0)),
            "pqos_misses":  float(self.cfg.train.ddpo.get("pqos_misses_reward_weight", 0.0)),
        }
        distance = str(self.cfg.train.ddpo.get("windowed_dtw_distance", "l1"))
        rewards_list, dtw_diag = [], []
        for i in range(N):
            if measured[i] is None:
                rewards_list.append(0.0)
                dtw_diag.append(None)
                continue
            gt_pc_i = (raw_gt_percore[i] if use_percore and raw_gt_percore is not None
                       else None)
            gt_pq_i = (raw_gt_pqos[i] if raw_gt_pqos is not None else None)
            d = dtw_reward_per_sample(measured[i], raw_gt_4d[i],
                                       self._windowed_metric_max, window_size=W,
                                       gt_percore=gt_pc_i, gt_pqos=gt_pq_i,
                                       weights=reward_weights,
                                       distance=distance)
            rewards_list.append(-float(d["mean_dtw"]))     # minimize DTW → negative reward
            dtw_diag.append(d)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device="cuda")

        # Logging summary
        ok = [d for d in dtw_diag if d is not None]
        if ok:
            log_keys = ["cpu_dtw", "io_dtw", "llc_dtw", "bw_dtw", "mean_dtw"]
            if use_percore:
                log_keys.append("per_core_cpu_dtw")
            # pqos diagnostics — logged whether weights are 0 or not, so you can see
            # the measured drift even before turning them on as reward.
            log_keys += ["pqos_ipc_dtw", "pqos_llc_dtw", "pqos_misses_dtw"]
            for k in log_keys:
                vals = [d[k] for d in ok if not np.isnan(d.get(k, float("nan")))]
                if not vals: continue
                v = float(np.mean(vals))
                self.log(f"win_dtw/{k.replace('_dtw','')}", v, prog_bar=(k=='mean_dtw'),
                         logger=True, on_epoch=True)
        self.log("win_dtw/n_ok", float(len(ok)), prog_bar=False, logger=True, on_epoch=True)

        # ---- (5) push to replay buffer (per window step, per diffusion step)
        T_diff = chains_per_step[0].shape[0] - 1   # n_timesteps
        for i in range(N):
            for w in range(W):
                trajectory = chains_per_step[w][:, i]     # (T_diff+1, S, T_a)
                log_prob   = log_probs_per_step[w][:, i]  # (T_diff,)
                cond_w     = metric_seq[i, w]              # (23,)
                # prev_state for this window step (scaled [-1,+1]); first step = idle
                if w == 0:
                    prev_state_w = -torch.ones_like(trajectory[0])
                else:
                    prev_state_w = chains_per_step[w - 1][-1, i]
                for t in range(1, T_diff):
                    self.replay_buffer.add_to_buffer(
                        state=(trajectory[t + 1], torch.tensor(t), cond_w, prev_state_w),
                        action=trajectory[t],
                        reward=rewards[i],
                        log_probs=log_prob[t],
                        final_state=trajectory[0],
                    )

        _mean_dtw_str = f"{float(np.mean([d['mean_dtw'] for d in ok])):.4f}" if ok else "n/a"
        print(f"[windowed-dtw] step={self.global_step} epoch={self.current_epoch} "
              f"N={N} W={W} mean_reward={float(rewards.mean()):.4f} mean_dtw={_mean_dtw_str}")

    def on_train_epoch_start(self) -> None:
        if not self.cfg.train.use_rl:
            return

        if self.cfg.train.ddpo.reward_type == "transformer":
            if self.trace_predictor_model is None:
                self.load_trace_predictor_model(self.cfg)

        if self.current_epoch % self.cfg.train.ddpo.num_inner_epochs != 0:
            return

        # Windowed-DTW RL path (each "sample" = W-sec consecutive window from K-means
        # medoid; rollout 5 actions auto-regressively, deploy as chained plan, reward =
        # mean per-family normalized DTW vs GT 5-sec metric).
        if bool(self.cfg.train.ddpo.get("windowed_dtw", False)):
            self._on_train_epoch_start_windowed()
            return

        self.trainer.datamodule.train_dataloader().dataset.shuffle_data(
            self.cfg.train.ddpo.num_batches_per_episode
        )
        self.replay_buffer.clear()

        merged_batch: dict = defaultdict(list)
        for batch_idx, batch in enumerate(self.trainer.datamodule.train_dataloader()):
            for key, value in batch.items():
                merged_batch[key].append(value)
            if batch_idx >= self.cfg.train.ddpo.num_batches_per_episode - 1:
                break

        stack_keys = {"clean_trace", "label", "prev_clean_trace", "prev_label"}
        # Also concat 1-D per-sample fields (medoid_id, training_data flag) to a flat tensor
        cat_keys = {"prev_medoid_id", "training_data"}
        merged_batch_tensor = {}
        for key, val in merged_batch.items():
            if key in stack_keys:
                merged_batch_tensor[key] = torch.stack(val).squeeze(1).to("cuda")
            elif key in cat_keys:
                merged_batch_tensor[key] = torch.cat([torch.as_tensor(v) for v in val]).to("cuda")
            else:
                merged_batch_tensor[key] = val

        rewards = self.sample_and_calculate_rewards(merged_batch_tensor)
        self.compute_advantage()
        self.trainer.logger.log_metrics(
            {"episode_reward": float(rewards.cpu().mean())},
            step=self.global_step,
        )
        self.samples = self.replay_buffer.sample()

    def compute_val(self, model, batch):
        return _compute_predictions(model, batch)

    def validation_step(self, batch, batch_idx):
        if self.num_epochs_trained == 0:
            print("Skipping validation step during initial sanity check...")
            return
        if batch_idx != 0 or not self.trainer.is_global_zero:
            return

        max_samples = 48
        trainer_errors, ema_errors = [], []
        mse_trainer_list, mse_ema_list = [], []

        for iter_idx in range(1):
            actions_trainer, mse_trainer, _ = self.compute_val(self.trainer_model, batch)
            actions_ema, mse_ema, _ = self.compute_val(self.ema_model, batch)
            mse_trainer_list.append(mse_trainer)
            mse_ema_list.append(mse_ema)

            for sample_idx, (action_trainer, action_ema) in enumerate(
                zip(actions_trainer, actions_ema)
            ):
                if sample_idx >= max_samples:
                    break
                print("Saving results for sample", sample_idx)
                dirname = os.path.join(
                    self.cfg.train.callbacks.checkpoint.dirpath,
                    f"step_{self.global_step}_batch_{sample_idx}",
                )
                if iter_idx == 0:
                    os.makedirs(dirname, exist_ok=True)
                    self.write_system_traces_to_file(dirname, batch["clean_trace"][sample_idx])
                    label_list = batch["label"][sample_idx].cpu()
                    prev_label_list = batch["prev_label"][sample_idx].cpu()
                    labels = self.write_labels_to_file(dirname, [prev_label_list, label_list])
                    trainer_errors.append(calculate_fleetbench_action_error(labels, action_trainer))
                    ema_errors.append(calculate_fleetbench_action_error(labels, action_ema))

                write_fleetbench_actions_to_h5_file(
                    [prev_label_list, action_trainer],
                    os.path.join(dirname, f"predicted_actions_trainer_{iter_idx}.h5"),
                )
                write_fleetbench_actions_to_h5_file(
                    [prev_label_list, action_ema],
                    os.path.join(dirname, f"predicted_actions_ema_{iter_idx}.h5"),
                )

        def _log(key, val, **kw):
            self.log(key, val, prog_bar=True, logger=True, on_epoch=True, **kw)

        _log("validation_loss/trainer_avg",    np.mean(mse_trainer_list))
        _log("validation_loss/ema_avg",        np.mean(mse_ema_list))
        _log("validation_loss/trainer_median", np.median(mse_trainer_list))
        _log("validation_loss/ema_median",     np.median(mse_ema_list))

        for prefix, errors in (
            ("trainer_action_error", trainer_errors),
            ("ema_action_error", ema_errors),
        ):
            _log(f"{prefix}/l1_mean",   np.mean(errors))
            _log(f"{prefix}/l1_median", np.median(errors))
            _log(f"{prefix}/l1_p90",    np.percentile(errors, 90))

        if self.cfg.train.get("skip_val_profiling", False):
            return

        async_validation = self.cfg.train.get("async_validation", False)
        profile_request = ProfileRequest(
            validation_data_path=self.cfg.train.callbacks.checkpoint.dirpath,
            my_destination_path=self.cfg.profiler.validation_destination_path,
            step=self.global_step,
            num_batches=max_samples,
            num_trials=1,
            logger=self.trainer.logger,
        )
        coro = self.profiler.profile(
            profile_request, self.cfg.data.trace_range,
            period=2, duration=18, aggregate_time_series=True,
        )
        if async_validation:
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            asyncio.run(coro)

    def on_before_zero_grad(self, optimizer):
        if self.global_rank == 0:
            self.ema.step_ema(self.ema_model, self.trainer_model)

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.trainer_model.parameters(), lr=self.cfg.train.optim.lr
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    dataloader = CustomDataLoader(
        cfg.data.data_path,
        cfg.data.test_data_path,
        max_time_steps=max_time_steps,
        batch_size=cfg.data.batch_size,
        test_batch_size=cfg.data.test_batch_size,
        use_rl=cfg.train.use_rl,
        ddpo_batch_size=cfg.train.ddpo.num_batches_per_episode if cfg.train.use_rl else None,
        aug_factor=cfg.data.get("aug_factor", 1),
        intra_only_aug=cfg.data.get("intra_only_aug", False),
        high_io_aug_factor=cfg.data.get("high_io_aug_factor", 1),
        io_raw_threshold=cfg.data.get("io_raw_threshold", 1000.0),
        cpu_avg_threshold=cfg.data.get("cpu_avg_threshold", 50.0),
        llc_max_threshold=cfg.data.get("llc_max_threshold", 100.0),
        bw_max_threshold=cfg.data.get("bw_max_threshold", 5.0),
        rl_train_ratio=cfg.data.get("rl_train_ratio", 0.333),
        rl_high_io_fraction=cfg.data.get("rl_high_io_fraction", 0.20),
        rl_high_cpu_fraction=cfg.data.get("rl_high_cpu_fraction", 0.20),
        rl_high_llc_fraction=cfg.data.get("rl_high_llc_fraction", 0.20),
        rl_high_bw_fraction=cfg.data.get("rl_high_bw_fraction", 0.20),
        rl_sample_method=cfg.data.get("rl_sample_method", "kmean"),
        rl_kmean_k=cfg.data.get("rl_kmean_k", 3000),
        rl_use_high_resource_supplements=cfg.data.get("rl_use_high_resource_supplements", False),
    )

    cfg.data.trace_range = {
        key: [float(v[0]), float(v[1])] for key, v in dataloader.trace_range.items()
    }

    arch = getattr(cfg, "model_arch", "unet")
    model = initialize_diffusion_model(cfg.model, model_arch=arch)
    wandb_logger = initialize_logger(cfg.log)
    callbacks = initialize_callbacks(cfg.train.callbacks)
    trainer_model = eval(cfg.train.trainer.trainer_model_name)(model, cfg)
    os.makedirs(cfg.train.callbacks.checkpoint.dirpath, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        devices=cfg.train.trainer.devices,
        accelerator=cfg.train.trainer.accelerator,
        strategy=cfg.train.trainer.strategy,
        precision=cfg.train.trainer.precision,
        log_every_n_steps=cfg.log.log_every_n_steps,
        gradient_clip_val=cfg.train.trainer.gradient_clip_val,
        check_val_every_n_epoch=cfg.train.trainer.check_val_every_n_epoch,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
    )

    ckpt_path = cfg.train.trainer.ckpt_path or None
    if cfg.train.run_train:
        print("Starting training...")
        trainer.fit(trainer_model, dataloader, ckpt_path=ckpt_path)
    if cfg.train.run_test:
        trainer.test(trainer_model, dataloader.test_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.set_float32_matmul_precision("highest")

    def _run_loop(lp: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(lp)
        lp.run_forever()

    thread = threading.Thread(target=_run_loop, args=(loop,))
    thread.start()

    train()

    pending = asyncio.all_tasks(loop)
    if pending:
        async def _gather():
            return await asyncio.gather(*pending)
        fut = asyncio.run_coroutine_threadsafe(_gather(), loop)
        print("Waiting for pending tasks to finish...")
        fut.result()
        print("All pending tasks finished.")

    loop.call_soon_threadsafe(loop.stop)
    thread.join()
