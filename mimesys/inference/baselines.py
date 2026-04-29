"""
Baseline generation methods for stress-ng action prediction.

All predictors operate on normalised trace vectors (float32, [−1, 1]) and
return action arrays of shape (STRESSORS, THREADS) with values in [0, 1].

No model weights or server state are required.
"""

import numpy as np
from scipy.spatial import cKDTree


class NearestNeighbor:
    """Return the action paired with the single closest training-set trace."""

    def __init__(self, train_traces: np.ndarray, train_actions: np.ndarray) -> None:
        """
        Parameters
        ----------
        train_traces : (N, trace_dim) float32, normalised to [−1, 1]
        train_actions: (N, STRESSORS, THREADS) float32, values in [0, 1]
        """
        self.tree    = cKDTree(train_traces)
        self.actions = train_actions

    def predict(self, trace_norm: np.ndarray) -> np.ndarray:
        """(trace_dim,) → (STRESSORS, THREADS)."""
        _, idx = self.tree.query(trace_norm, k=1)
        return self.actions[idx].copy()

    def predict_batch(self, traces_norm: np.ndarray) -> np.ndarray:
        """(T, trace_dim) → (T, STRESSORS, THREADS)."""
        _, idxs = self.tree.query(traces_norm, k=1)
        return self.actions[idxs].copy()


class LinearInterpolation:
    """Inverse-distance-weighted blend of the k nearest training actions."""

    def __init__(self, train_traces: np.ndarray, train_actions: np.ndarray,
                 k: int = 5) -> None:
        self.tree    = cKDTree(train_traces)
        self.actions = train_actions
        self.k       = k

    def predict(self, trace_norm: np.ndarray) -> np.ndarray:
        """(trace_dim,) → (STRESSORS, THREADS)."""
        k = min(self.k, len(self.actions))
        dists, idxs = self.tree.query(trace_norm, k=k)
        dists   = np.where(dists == 0, 1e-8, dists)
        weights = 1.0 / dists
        weights /= weights.sum()
        return np.sum(self.actions[idxs] * weights[:, None, None], axis=0)

    def predict_batch(self, traces_norm: np.ndarray) -> np.ndarray:
        """(T, trace_dim) → (T, STRESSORS, THREADS)."""
        return np.stack([self.predict(t) for t in traces_norm])


class SingleStressor:
    """
    Heuristic baseline: distribute per-core CPU utilisation across threads
    using a single stressor row.  No training data required.
    """

    def __init__(self, stressor_idx: int = 10,
                 n_stressors: int = 13, n_threads: int = 20) -> None:
        self.stressor_idx = stressor_idx
        self.n_stressors  = n_stressors
        self.n_threads    = n_threads

    @classmethod
    def from_actions(cls, train_actions: np.ndarray, stressor_idx: int = 10) -> "SingleStressor":
        """Infer n_stressors and n_threads from train_actions shape (N, STRESSORS, THREADS)."""
        _, n_stressors, n_threads = train_actions.shape
        return cls(stressor_idx=stressor_idx, n_stressors=n_stressors, n_threads=n_threads)

    def predict(self, trace_norm: np.ndarray) -> np.ndarray:
        """(trace_dim,) → (STRESSORS, THREADS).  First n_threads dims are CPU cores."""
        cpu_frac = ((trace_norm[:self.n_threads] + 1) / 2).clip(0, 1)
        action = np.zeros((self.n_stressors, self.n_threads), dtype=np.float32)
        action[self.stressor_idx] = cpu_frac
        return action

    def predict_batch(self, traces_norm: np.ndarray) -> np.ndarray:
        """(T, trace_dim) → (T, STRESSORS, THREADS)."""
        return np.stack([self.predict(t) for t in traces_norm])
