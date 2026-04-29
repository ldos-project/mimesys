"""
InferenceEngine
===============
Stateful inference engine: owns the loaded model, baseline predictors,
and trace normalisation ranges.  Decoupled from the HTTP layer so it can
be used standalone (e.g. in evaluation scripts) or embedded in a server.

Usage
-----
    from mimesys.inference.engine import InferenceEngine

    engine = InferenceEngine.from_checkpoint(
        ckpt_path="/path/to/last.ckpt",
        cfg=hydra_cfg,
        dataloader=custom_dl,
        device="cuda",
    )

    pred = engine.generate_step(
        trace_raw={"io": 3000, "l3_cache_usage_socket_0": 9000},
        method="diffusion",
    )
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from mimesys.inference.baselines import (
    NearestNeighbor, LinearInterpolation, SingleStressor,
)

# Action tensor dimensions: (STRESSORS, THREADS) = (13, 20)
ACTION_STRESSORS: int = 13
ACTION_THREADS:   int = 20

# ── Trace layout ─────────────────────────────────────────────────────────────
METRIC_KEYS: list[str] = [
    *[f"avg_cpu_utilizations_core_{i:02d}" for i in range(20)],
    "io",
    "l3_cache_usage_socket_0",
    "l3_cache_usage_socket_1",
    "memory_bandwidth_socket_0",
    "memory_bandwidth_socket_1",
]

METRIC_UNITS: dict[str, str] = {
    **{f"avg_cpu_utilizations_core_{i:02d}": "%" for i in range(20)},
    "io":                        "KB/s",
    "l3_cache_usage_socket_0":   "MB",
    "l3_cache_usage_socket_1":   "MB",
    "memory_bandwidth_socket_0": "GB/s",
    "memory_bandwidth_socket_1": "GB/s",
}

STRESSOR_NAMES: list[str] = [
    "brk", "dev-shm", "env", "fallocate", "llc-affinity",
    "mmapfixed", "mmaphuge", "readahead", "seek", "stream",
    "cpu", "vm", "io",
]

METHOD_NAMES = ("diffusion", "nearest_neighbor", "linear_interpolation", "single_stressor")


class InferenceEngine:
    """
    Bundles the diffusion model, baseline predictors, and normalisation state.

    Parameters are set via :meth:`from_checkpoint` rather than ``__init__``
    to keep construction explicit.
    """

    def __init__(self) -> None:
        self.model:        Optional[torch.nn.Module] = None
        self.device:       str  = "cpu"
        self.ckpt_meta:    dict = {}
        self.trace_range:  dict = {}
        self.nn:           Optional[NearestNeighbor]      = None
        self.li:           Optional[LinearInterpolation]  = None
        self.ss:           Optional[SingleStressor]       = None

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        cfg,
        dataloader,
        device: str = "cuda",
    ) -> "InferenceEngine":
        """
        Load a diffusion model checkpoint and build baseline indices.

        Parameters
        ----------
        ckpt_path  : absolute path to .ckpt file
        cfg        : Hydra OmegaConf config (needs cfg.model)
        dataloader : CustomDataLoader instance (val_dataset used for baselines)
        device     : "cuda" or "cpu"
        """
        from mimesys.training.utils import initialize_diffusion_model

        engine = cls()
        engine.device      = device
        engine.trace_range = dataloader.trace_range

        # Build baseline predictors from val split
        engine._build_baselines(dataloader)

        # Load model
        model = initialize_diffusion_model(cfg.model, model_arch="unet").to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)

        if "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
            weight_source = "ema"
        elif "state_dict" in ckpt:
            state = {
                k.replace("trainer_model.", ""): v
                for k, v in ckpt["state_dict"].items()
                if "trainer_model." in k
            }
            model.load_state_dict(state)
            weight_source = "trainer"
        else:
            model.load_state_dict(ckpt)
            weight_source = "direct"

        model.eval()
        engine.model     = model
        engine.ckpt_meta = {
            "path":         ckpt_path,
            "epoch":        int(ckpt.get("epoch", -1)),
            "global_step":  int(ckpt.get("global_step", -1)),
            "weight_source": weight_source,
        }
        return engine

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize_trace(self, raw: dict[str, float]) -> np.ndarray:
        """
        Convert ``{metric_name: raw_value}`` → normalised (trace_dim,) float32 in [−1, 1].
        Missing keys default to 0; values outside the training range are clipped to ±1.
        """
        vec = []
        for key in METRIC_KEYS:
            raw_val = raw.get(key, 0.0)
            lo, hi  = (float(self.trace_range[key][0]),
                       float(self.trace_range[key][1])) if key in self.trace_range else (0.0, 1.0)
            if hi > lo:
                norm = max(-1.0, min(1.0, (raw_val - lo) / (hi - lo) * 2 - 1))
            else:
                norm = 0.0
            vec.append(norm)
        return np.array(vec, dtype=np.float32)

    # ── Core generation ───────────────────────────────────────────────────────

    def generate_step(
        self,
        trace_raw:    dict[str, float],
        prev_action:  Optional[list[list[float]]] = None,
        method:       str   = "diffusion",
        n_chains:     int   = 3,
        cfg_guide_w:  float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a single action from one resource-usage snapshot.

        Returns
        -------
        prev_arr : (STRESSORS, THREADS) float32, the decoded prev_action (zeros if None)
        pred     : (STRESSORS, THREADS) float32 in [0, 1]
        """
        trace_norm = self.normalize_trace(trace_raw)

        if method == "diffusion":
            prev_t   = self._encode_prev_action(prev_action)
            pred     = self._diffusion_infer(trace_norm, prev_t, n_chains, cfg_guide_w)
            prev_arr = ((prev_t.numpy() + 1) / 2.0).clip(0, 1)

        elif method == "nearest_neighbor":
            pred     = self.nn.predict(trace_norm)
            prev_arr = np.zeros((ACTION_STRESSORS, ACTION_THREADS), dtype=np.float32)

        elif method == "linear_interpolation":
            pred     = self.li.predict(trace_norm)
            prev_arr = np.zeros((ACTION_STRESSORS, ACTION_THREADS), dtype=np.float32)

        elif method == "single_stressor":
            pred     = self.ss.predict(trace_norm)
            prev_arr = np.zeros((ACTION_STRESSORS, ACTION_THREADS), dtype=np.float32)

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: {METHOD_NAMES}")

        return prev_arr, pred

    def generate_series(
        self,
        traces:              list[dict[str, float]],
        method:              str   = "diffusion",
        initial_prev_action: Optional[list[list[float]]] = None,
        n_chains:            int   = 3,
        cfg_guide_w:         float = 3.0,
    ) -> list[np.ndarray]:
        """
        Generate a sequence of actions from an ordered list of traces.

        For the diffusion method the action at step *t* is fed as ``prev_action``
        at step *t+1* (autoregressive).  Baselines process each step independently.

        Returns
        -------
        List of T arrays, each (STRESSORS, THREADS) float32 in [0, 1].
        """
        if method in ("nearest_neighbor", "linear_interpolation", "single_stressor"):
            traces_norm = np.stack([self.normalize_trace(t) for t in traces])
            if method == "nearest_neighbor":
                return list(self.nn.predict_batch(traces_norm))
            if method == "linear_interpolation":
                return list(self.li.predict_batch(traces_norm))
            return list(self.ss.predict_batch(traces_norm))

        # Diffusion — autoregressive
        prev_action = initial_prev_action
        preds       = []
        for trace_raw in traces:
            _, pred = self.generate_step(trace_raw, prev_action, method, n_chains, cfg_guide_w)
            preds.append(pred)
            prev_action = pred.tolist()
        return preds

    def generate_series_parallel_refine(
        self,
        traces:      list[dict[str, float]],
        n_chains:    int   = 3,
        cfg_guide_w: float = 3.0,
    ) -> list[np.ndarray]:
        """
        Two-pass parallel generation — fully batched.

        Pass 1 — all T steps in one batched call with prev_action=None:
            [m1, None] → a1,  [m2, None] → a2,  ...,  [mT, None] → aT

        Pass 2 — all T steps in one batched call using the pass-1 predecessor:
            a1' = a1  (no predecessor for step 0)
            [m2, a1] → a2',  [m3, a2] → a3',  ...,  [mT, a_{T-1}] → aT'

        Both passes run as a single p_sample_loop(batch_size=T) call, so the
        total cost is 2 × n_chains denoising sweeps instead of 2×T.

        Returns the pass-2 predictions.
        """
        T = len(traces)

        # Normalise all traces up front
        trace_norms = np.stack([self.normalize_trace(tr) for tr in traces])  # (T, trace_dim)

        # ── Pass 1: all prev = -1 (None) ──────────────────────────────────────
        prev_none = -torch.ones(T, ACTION_STRESSORS, ACTION_THREADS, dtype=torch.float32)
        pass1_arr = self._diffusion_infer_batch(trace_norms, prev_none, n_chains, cfg_guide_w)
        # pass1_arr: (T, STRESSORS, THREADS) in [0, 1]

        # ── Pass 2: prev[t] = pass1[t-1], keep pass1[0] for step 0 ───────────
        # Build prev tensor: slot 0 stays -1 (no predecessor), slots 1..T-1 use pass1[0..T-2]
        prev_pass2 = -torch.ones(T, ACTION_STRESSORS, ACTION_THREADS, dtype=torch.float32)
        if T > 1:
            # Convert pass1[0..T-2] from [0,1] to model space [-1,1]
            prev_pass2[1:] = torch.tensor(pass1_arr[:-1] * 2 - 1, dtype=torch.float32)

        pass2_arr = self._diffusion_infer_batch(trace_norms, prev_pass2, n_chains, cfg_guide_w)
        # step 0 had prev=-1 in both passes; override with pass1[0] for consistency
        pass2_arr[0] = pass1_arr[0]

        return [pass2_arr[t] for t in range(T)]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_prev_action(
        self, prev_action: Optional[list[list[float]]]
    ) -> torch.Tensor:
        """Convert [0,1]-valued list-of-lists → model-space tensor (−1 if None)."""
        if prev_action is None:
            return -torch.ones(ACTION_STRESSORS, ACTION_THREADS, dtype=torch.float32)
        arr = np.array(prev_action, dtype=np.float32)
        if arr.shape != (ACTION_STRESSORS, ACTION_THREADS):
            raise ValueError(
                f"prev_action must be ({ACTION_STRESSORS}, {ACTION_THREADS}), got {arr.shape}"
            )
        return torch.tensor(arr * 2 - 1, dtype=torch.float32)

    def _diffusion_infer(
        self,
        trace_norm:  np.ndarray,
        prev_t:      torch.Tensor,
        n_chains:    int,
        cfg_guide_w: float,
    ) -> np.ndarray:
        """Run n_chains diffusion forward passes and return their mean."""
        self.model.cfg_guide_w = cfg_guide_w
        shape = (1, ACTION_STRESSORS, ACTION_THREADS)
        with torch.no_grad():
            preds = []
            for _ in range(n_chains):
                chain = self.model.p_sample_loop(shape, {
                    "metric":      torch.tensor(trace_norm).unsqueeze(0).to(self.device),
                    "prev_action": prev_t.unsqueeze(0).to(self.device),
                })
                preds.append(chain[-1].cpu())
        pred = torch.stack(preds).mean(0).squeeze(0)
        return ((pred.numpy() + 1) / 2.0).clip(0, 1)

    def _diffusion_infer_batch(
        self,
        trace_norms: np.ndarray,   # (T, trace_dim)
        prev_ts:     torch.Tensor, # (T, ACTION_STRESSORS, ACTION_THREADS)
        n_chains:    int,
        cfg_guide_w: float,
        chunk_size:  int = 32,
    ) -> np.ndarray:
        """
        Batched diffusion inference over T steps, processed in chunks to
        avoid OOM.  Runs n_chains passes per chunk and averages the results.

        Returns
        -------
        np.ndarray of shape (T, ACTION_STRESSORS, ACTION_THREADS) in [0, 1].
        """
        T = trace_norms.shape[0]
        self.model.cfg_guide_w = cfg_guide_w
        results = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            metric_t   = torch.tensor(
                trace_norms[start:end], dtype=torch.float32
            ).to(self.device)
            prev_t_dev = prev_ts[start:end].to(self.device)
            B = end - start
            shape = (B, ACTION_STRESSORS, ACTION_THREADS)
            with torch.no_grad():
                preds = []
                for _ in range(n_chains):
                    chain = self.model.p_sample_loop(shape, {
                        "metric":      metric_t,
                        "prev_action": prev_t_dev,
                    })
                    preds.append(chain[-1].cpu())   # (B, STRESSORS, THREADS)
            chunk_pred = torch.stack(preds).mean(0) # (B, STRESSORS, THREADS)
            results.append(chunk_pred)
        pred = torch.cat(results, dim=0)            # (T, STRESSORS, THREADS)
        return ((pred.numpy() + 1) / 2.0).clip(0, 1)

    def _build_baselines(self, dataloader) -> None:
        """Build cKDTree-backed predictors from the validation split."""
        traces  = []
        actions = []
        for item in dataloader.val_dataset:
            traces.append(item["clean_trace"].numpy())
            actions.append(((item["label"].numpy() + 1) / 2).clip(0, 1))
        train_t = np.stack(traces).astype(np.float32)
        train_a = np.stack(actions).astype(np.float32)

        self.nn = NearestNeighbor(train_t, train_a)
        self.li = LinearInterpolation(train_t, train_a, k=5)
        self.ss = SingleStressor()
