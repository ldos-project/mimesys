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
        prev_medoid_n = int(self.cfg.train.ddpo.get("prev_medoid_n", 1))
        if prev_medoid_n > 1:
            # Sample K candidates per row in a single inflated-batch diffusion call,
            # then pick the per-row medoid (sample minimising sum-of-L1 to others).
            # This reduces per-step prev-state variance vs single-sample inference.
            K = prev_medoid_n
            prev_trace_K   = prev_trace.repeat_interleave(K, dim=0).cuda()    # (K*N, dim)
            prev_states_K  = prev_states.repeat_interleave(K, dim=0)           # (K*N, S, T)
            shape_K        = (K * N,) + tuple(x0.shape[1:])
            with torch.no_grad():
                chains_K, _ = self.trainer_model.p_sample_loop_with_logprobs(
                    shape_K, {"metric": prev_trace_K, "prev_action": prev_states_K}
                )
            final = chains_K[-1].view(N, K, *x0.shape[1:])     # (N, K, S, T)
            final_np = final.detach().cpu().numpy()
            S = x0.shape[1]; Tth = x0.shape[2]
            import numpy as _np
            medoid_out = _np.empty((N, S, Tth), dtype=_np.float32)
            for _i in range(N):
                F = final_np[_i].reshape(K, -1)
                sd = _np.array([float(_np.abs(F - F[k:k+1]).mean(axis=1).sum())
                                for k in range(K)])
                medoid_out[_i] = final_np[_i, int(_np.argmin(sd))]
            prev_states = torch.tensor(medoid_out, dtype=x0.dtype,
                                       device=x0.device).cuda()
        else:
            chains, _ = self.trainer_model.p_sample_loop_with_logprobs(
                x0.shape, {"metric": prev_trace, "prev_action": prev_states}
            )
            prev_states = chains[-1]
        for idx in range(N):
            if is_training_data[idx]:
                prev_states[idx] = prev_labels[idx]

        trace = (trace + torch.randn_like(trace) * 1e-4).cuda()

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
                    write_fleetbench_actions_to_h5_file(
                        [prev_state, action_k],
                        os.path.join(dirname, "predicted_actions_trainer_0.h5"),
                    )

            profile_request = ProfileRequest(
                validation_data_path=f"{checkpoint_dir}/{exp_path}",
                my_destination_path=self.cfg.profiler.destination_path,
                step=self.global_step,
                num_batches=N * best_of_n,
                num_trials=1,
                logger=self.trainer.logger,
                model_type="trainer",
                io_reward_weight=self.cfg.train.ddpo.get("io_reward_weight", 1.0),
                low_resource_penalty_weight=self.cfg.train.ddpo.get("low_resource_penalty_weight", 0.0),
                membw_reward_weight=self.cfg.train.ddpo.get("membw_reward_weight", 1.0),
                llc_reward_weight=self.cfg.train.ddpo.get("llc_reward_weight", 1.0),
            )
            print("Profile request:", profile_request)
            _, avg_emd_by_batch, metrics_std_by_batch = asyncio.run(
                self.profiler.profile(
                    profile_request, self.cfg.data.trace_range,
                    period=2, duration=6, aggregate_time_series=True,
                )
            )

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

    def on_train_epoch_start(self) -> None:
        if not self.cfg.train.use_rl:
            return

        if self.cfg.train.ddpo.reward_type == "transformer":
            if self.trace_predictor_model is None:
                self.load_trace_predictor_model(self.cfg)

        if self.current_epoch % self.cfg.train.ddpo.num_inner_epochs != 0:
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

        stack_keys = {"clean_trace", "label", "prev_clean_trace"}
        merged_batch_tensor = {
            key: (torch.stack(val).squeeze(1).to("cuda") if key in stack_keys else val)
            for key, val in merged_batch.items()
        }

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
    )

    cfg.data.trace_range = {
        key: [float(v[0]), float(v[1])] for key, v in dataloader.trace_range.items()
    }

    model = initialize_diffusion_model(cfg.model, model_arch="unet")
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
