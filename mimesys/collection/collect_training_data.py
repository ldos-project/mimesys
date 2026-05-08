"""
collect_surrogate_v2.py
=======================
Active-learning data collection with 13-action space
(10 non-IO + Readahead[10] + Fallocate_4MB[11] + Hdd_1MB[12]).

Round 0  : initial_candidates  — one-hot style sweep covering each action
           in isolation at various thread counts and weight scales (~1300 plans).
Rounds 1+: hull:fps = 5:5 (io mutation disabled)
           - 50 % from hull interpolation  }
           - 50 % from fps novelty          } via propose_by_hull_mixed_fps_hybrid (hull_fps_ratio=1)

Output:    ~/mimesys_training_data/surrogate_v2

Usage:
  cd mimesys/collection/scripts
  python collect_surrogate_v2.py [--rounds 50] [--restart]
  # --rounds  : active-learning rounds after round 0 (default: 50)
  # --restart : delete any existing rounds and start fresh from round 0
"""

import argparse
import asyncio
import os
import random
import re
import shutil
import sys
import h5py

import numpy as np

from mimesys.collection.profiling_server import InitializeRequest, Profiler
from mimesys.schema.machine import Machine

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from mimesys.schema.machine import Machine
from scipy.spatial import Delaunay

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_PATH = os.path.expanduser("~/mimesys_training_data/surrogate_v2")

_DEFAULT_PROFILING_MACHINES = []

# Allow overriding the worker pool from the environment so we can target a
# subset (e.g. one c220g5 host for a smoke test) without editing the file.
_env_machines = os.environ.get("MIMESYS_PROFILING_MACHINES", "").strip()
PROFILING_MACHINES = (
    [h.strip() for h in _env_machines.split(",") if h.strip()]
    if _env_machines else _DEFAULT_PROFILING_MACHINES
)

# SSH credentials — override via env vars for CI / different users
SSH_USER     = os.environ.get("MIMESYS_SSH_USER",     "dhkim")
SSH_KEY_PATH = os.environ.get("MIMESYS_SSH_KEY",      os.path.expanduser("~/.ssh/id_rsa_utns"))
MY_HOSTNAME  = os.environ.get("MIMESYS_MY_HOSTNAME",  "mew3")

NUM_ACTIONS    = 13
NUM_THREADS    = 20
_PER_MACHINE_BATCH = int(os.environ.get("MIMESYS_PER_MACHINE_BATCH", "16"))
BATCH_SIZE     = _PER_MACHINE_BATCH * len(PROFILING_MACHINES)   # active-learning round size
NUM_METRICS    = 26

# 3:7:0 ratio parameters (30% hull, 70% fps, no IO mutation)
HULL_FPS_RATIO = 3/7     # hull:fps split inside hull_mixed_fps_hybrid (3/7 → 30/70)


def sample_scaling_weight():
    weights = np.arange(90, 105, 5)
    probabilities = weights / weights.sum()
    return np.random.choice(weights, p=probabilities)

def propose_candidates_by_random(timestamp=2, num_actions=13, num_threads=20, n_candidates=1000):
    candidates = []
    action_shape = (timestamp, num_threads, num_actions)

    def zero_out_random_rows(mutated):
        max_rows_to_zero = max(1, len(mutated) - 1)
        num_rows_to_zero = random.randint(1, max_rows_to_zero)
        rows_to_zero = random.sample(range(len(mutated)), num_rows_to_zero)
        for row_idx in rows_to_zero:
            mutated[row_idx] = [0.0] * len(mutated[row_idx])
        return mutated

    for _ in range(n_candidates):
        candidate = []
        for _ in range(action_shape[0]):
            thread = []
            for _ in range(action_shape[1]):
                num_cols_to_zero = random.randint(action_shape[2] // 2, action_shape[2] - 1)
                action = [random.uniform(0.0, 1.0) for _ in range(action_shape[2])]
                for _ in range(num_cols_to_zero):
                    zero_idx = random.randint(0, action_shape[2] - 1)
                    action[zero_idx] = 0.0
                if sum(action) > 1:
                    action = [v / sum(action) for v in action]
                scale = sample_scaling_weight() / 100.0
                action = [v * scale for v in action]
                thread.append(action)
            thread = zero_out_random_rows(thread)
            candidate.append(thread)
        candidates.append(candidate)

    return candidates


def propose_candidates_by_pool_mutation(A, n_candidates=1000, noise_std=0.03):
    """
    Generate novel action candidates by mutating actions drawn from pool A.

    Four mutation operators are applied with equal probability:

      thread_swap     – swap a random subset of thread rows between two parents.
                        The result takes some threads from parent-1 and the rest
                        from parent-2, producing a new thread-level mixture.

      stressor_crossover – swap a random subset of stressor *columns* between two
                        parents.  For each selected stressor index k every thread's
                        k-th weight is taken from parent-2 instead of parent-1.

      blend           – convex combination of 2–4 parents drawn with Dirichlet
                        weights.  Per-thread sums are renormalised to ≤ 1 after
                        blending.

      scale_noise     – perturb a single parent's weights with multiplicative
                        log-normal noise (std=noise_std), then re-sparsify by
                        zeroing threads whose total activity falls below a random
                        threshold.

    Action shape: [n_timestamps][n_threads][n_stressors].
    Only timestamp index 1 (curr_action) is mutated; timestamp 0 (prev_action) is
    copied verbatim from the primary parent.

    Returns a list of n_candidates new actions (same nested-list format as A).
    """
    if not A:
        raise ValueError("Pool A is empty; cannot mutate.")

    def _norm_thread(t):
        s = sum(t)
        return [v / s for v in t] if s > 1.0 else list(t)

    def _action_as_arrays(a):
        return [np.array(ts, dtype=float) for ts in a]

    def _arrays_to_action(arrs):
        return [ts.tolist() for ts in arrs]

    def _apply_to_curr(arrs, fn):
        ts_idx = 1 if len(arrs) > 1 else 0
        result = [arrs[0].copy()]                  # keep prev_action unchanged
        curr = fn(arrs[ts_idx].copy())             # mutate curr_action (ts index 1)
        if ts_idx == 0:
            result[0] = curr
        else:
            result.append(curr)
        for ts in arrs[2:]:                        # keep any extra timestamps
            result.append(ts.copy())
        return result

    # ── operator 1: swap a random subset of thread rows ──────────────────────
    def _thread_swap(a1_arrs, a2_arrs):
        def op(curr):                              # curr shape: (n_threads, n_stressors)
            n_th = curr.shape[0]
            swap_idx = np.random.choice(n_th,
                                        size=random.randint(1, max(1, n_th // 2)),
                                        replace=False)
            c2 = a2_arrs[1 if len(a2_arrs) > 1 else 0]
            result = curr.copy()
            for i in swap_idx:
                if i < c2.shape[0]:
                    result[i] = c2[i].copy()
            return result
        return _apply_to_curr(a1_arrs, op)

    # ── operator 2: swap a random subset of stressor columns ─────────────────
    def _stressor_crossover(a1_arrs, a2_arrs):
        def op(curr):
            n_str = curr.shape[1]
            k = random.randint(1, max(1, n_str // 3))
            col_idx = np.random.choice(n_str, size=k, replace=False)
            c2 = a2_arrs[1 if len(a2_arrs) > 1 else 0]
            result = curr.copy()
            for c in col_idx:
                result[:, c] = c2[:min(curr.shape[0], c2.shape[0]), c]
            # renormalise threads whose sum now exceeds 1
            for i in range(result.shape[0]):
                result[i] = np.array(_norm_thread(result[i].tolist()))
            return result
        return _apply_to_curr(a1_arrs, op)

    # ── operator 3: blend 2–4 parents with Dirichlet weights ─────────────────
    def _blend(parents_arrs):
        k = len(parents_arrs)
        weights = np.random.dirichlet(np.ones(k))

        def op(idx):
            result = sum(w * p[idx] for w, p in zip(weights, parents_arrs))
            for i in range(result.shape[0]):
                result[i] = np.array(_norm_thread(result[i].tolist()))
            return result

        ts_idx = 1 if all(len(p) > 1 for p in parents_arrs) else 0
        out = [parents_arrs[0][0].copy()]          # prev_action from first parent
        blended = op(ts_idx)
        if ts_idx == 0:
            out[0] = blended
        else:
            out.append(blended)
        for extra_idx in range(2, len(parents_arrs[0])):
            out.append(op(extra_idx))
        return out

    # ── operator 4: multiplicative log-normal noise + random thread zeroing ──
    def _scale_noise(a_arrs):
        def op(curr):
            result = curr * np.exp(np.random.randn(*curr.shape) * noise_std)
            # zero out a random fraction of threads to maintain sparsity
            n_th = result.shape[0]
            n_zero = random.randint(0, max(0, n_th // 4))
            if n_zero:
                zero_rows = np.random.choice(n_th, size=n_zero, replace=False)
                result[zero_rows] = 0.0
            for i in range(result.shape[0]):
                result[i] = np.array(_norm_thread(result[i].tolist()))
            return result
        return _apply_to_curr(a_arrs, op)

    # ── operator 5: socket crossover — S0 from parent-1, S1 from parent-2 ────
    def _socket_crossover(a1_arrs, a2_arrs):
        """Take threads 0–9 from parent-1 and threads 10–19 from parent-2,
        guaranteeing a socket-level load asymmetry in every produced action."""
        def op(curr):
            n_th  = curr.shape[0]
            split = n_th // 2          # 10 for the standard 20-thread layout
            c2    = a2_arrs[1 if len(a2_arrs) > 1 else 0]
            result = curr.copy()
            result[split:] = c2[split:min(n_th, c2.shape[0])].copy()
            return result
        return _apply_to_curr(a1_arrs, op)

    # ── operator 6: socket scale — independent per-socket intensity scaling ──
    def _socket_scale(a_arrs):
        """Apply independent log-normal scale factors to the two socket halves,
        creating a wide range of S0:S1 load ratios while preserving stressor
        structure within each socket."""
        socket_noise_std = max(noise_std * 5, 0.3)   # wider than per-element noise
        def op(curr):
            n_th  = curr.shape[0]
            split = n_th // 2
            result = curr.copy()
            for slc in [slice(0, split), slice(split, n_th)]:
                scale = np.exp(np.random.randn() * socket_noise_std)
                result[slc] = result[slc] * scale
            for i in range(result.shape[0]):
                result[i] = np.array(_norm_thread(result[i].tolist()))
            return result
        return _apply_to_curr(a_arrs, op)

    operators = ["thread_swap", "stressor_crossover", "blend", "scale_noise",
                 "socket_crossover", "socket_scale"]
    candidates = []

    for _ in range(n_candidates):
        op = random.choice(operators)

        if op == "blend":
            k = random.randint(2, min(4, len(A)))
            parents = random.choices(A, k=k)
            parents_arrs = [_action_as_arrays(p) for p in parents]
            result_arrs = _blend(parents_arrs)
        else:
            p1, p2 = random.choices(A, k=2)
            a1 = _action_as_arrays(p1)
            a2 = _action_as_arrays(p2)
            if op == "thread_swap":
                result_arrs = _thread_swap(a1, a2)
            elif op == "stressor_crossover":
                result_arrs = _stressor_crossover(a1, a2)
            elif op == "socket_crossover":
                result_arrs = _socket_crossover(a1, a2)
            elif op == "socket_scale":
                result_arrs = _socket_scale(a1)
            else:  # scale_noise
                result_arrs = _scale_noise(a1)

        candidates.append(_arrays_to_action(result_arrs))

    return candidates


# M layout: [prev_metrics(NUM_METRICS), curr_metrics(NUM_METRICS), noop_metrics(NUM_METRICS)] = NUM_METRICS * 3 dims total.
# M ordering verified: group 0 = prev_action, group 1 = curr_action, group 2 = no-op.
# e.g., Within each 26-dim group the last 6 are the key metrics in this order:
#   idx+20  avg_cpu_utilizations_total
#   idx+21  io
#   idx+22  l3_cache_usage_socket_0
#   idx+23  l3_cache_usage_socket_1
#   idx+24  memory_bandwidth_socket_0
#   idx+25  memory_bandwidth_socket_1
# For curr_action (group 1, offset 26): flat indices 46–51.
_KEY_METRIC_INDICES = [
        NUM_METRICS * 2 - 6,
        NUM_METRICS * 2 - 5,
        NUM_METRICS * 2 - 4,
        NUM_METRICS * 2 - 3,
        NUM_METRICS * 2 - 2,
        NUM_METRICS * 2 - 1
    ]


def _flatten_action_for_knn(a):
    """Sum stressor weights across threads for the curr_action timestamp (index 1) → 13-D vector.

    Each action a has shape [timestamps][threads][stressors].
    We use only timestamp 1 (curr_action) because we want k-NN similarity based on
    the action that drives group-1 (curr_action) metrics we are diversifying.
    Falls back to summing all timestamps if the action has fewer than 2 timestamps.
    """
    n_stressors = len(a[0][0])
    ts_idx = 1 if len(a) > 1 else 0
    arr = np.zeros(n_stressors)
    for thread in a[ts_idx]:
        arr += np.array(thread)
    return arr


def propose_by_hull_mixed_fps_hybrid(A, M, n_candidates, grid_bins=10, k_neighbors=3,
                                      n_estimators=100, fps_oversample=3,
                                      hull_fps_ratio=1):
    """
    Hybrid of hull_mixed and fps_rf with configurable pool ratio and random
    final selection.

    Steps:
      1. Build hull pool via convex-hull interpolation across groups
         [(0,2,4),(0,3,5),(0,5),(2,3),(4,5)]. Let N = hull pool size.
      2. Train RF surrogate (action → metrics). Generate fps_n * fps_oversample
         candidates — a mix of pool mutations (mutation_ratio) and fresh random
         samples (1 - mutation_ratio) — predict metrics, then greedily select
         fps_n most novel via greedy FPS.
      3. Merge hull pool (N) + fps pool (fps_n) → total pool.
      4. Randomly sample n_candidates from the merged pool.

    hull_fps_ratio=1  → equal pools (1:1); hull_fps_ratio=9 → 9:1 hull:fps.
    mutation_ratio=0.5 → half the fps oversample pool comes from pool mutations,
                         half from fresh random candidates.
    Metric indices: 0=CPU%  1=IO  2=LLC-S0  3=LLC-S1  4=BW-S0  5=BW-S1
                    6=CPU-S0  7=CPU-S1  (per-socket CPU util, cols 26-35 / 36-45)
    """
    from collections import Counter

    M_key6 = M[:, _KEY_METRIC_INDICES].astype(float)
    # Per-socket CPU util: mean across the 10 threads per socket
    cpu_s0 = M[:, NUM_ACTIONS:NUM_ACTIONS+10].astype(float).mean(axis=1, keepdims=True)
    cpu_s1 = M[:, NUM_ACTIONS+10:NUM_ACTIONS+20].astype(float).mean(axis=1, keepdims=True)
    M_key  = np.hstack([M_key6, cpu_s0, cpu_s1])          # (N, 8)

    # ── 1. Build hull pool ────────────────────────────────────────────────────
    GROUPS = [
        (0, 2, 4),
        (0, 3, 5),
        (0, 5),
        (2, 3),
        (4, 5),
        (6, 7),    # cpu-s0 vs cpu-s1 (socket CPU asymmetry)
    ]

    n_str   = len(A[0][0][0])
    hull_pool = []

    for dims in GROUPS:
        d = len(dims)
        M_sub = M_key[:, list(dims)]
        mn, mx = M_sub.min(0), M_sub.max(0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        M_norm = (M_sub - mn) / rng

        try:
            tri = Delaunay(M_norm)
        except Exception:
            print(f"  [hull_fps_hybrid] dims={dims}: Delaunay failed, skipping")
            continue

        g = np.linspace(0.5 / grid_bins, 1.0 - 0.5 / grid_bins, grid_bins)
        grids = np.meshgrid(*[g] * d, indexing='ij')
        centres = np.stack([gg.ravel() for gg in grids], axis=1)

        inside_mask = tri.find_simplex(centres) >= 0
        inside_centres = centres[inside_mask]
        if len(inside_centres) == 0:
            continue

        def cell_of(pts):
            return np.clip((pts * grid_bins).astype(int), 0, grid_bins - 1)

        cell_counts = Counter(map(tuple, cell_of(M_norm).tolist()))

        def priority(c):
            ci   = tuple(cell_of(c[None])[0].tolist())
            cnt  = cell_counts.get(ci, 0)
            dist = float(np.linalg.norm(M_norm - c, axis=1).min())
            return (cnt, -dist)

        sorted_centres = inside_centres[
            sorted(range(len(inside_centres)), key=lambda k: priority(inside_centres[k]))
        ]

        before = len(hull_pool)
        for pt in sorted_centres:
            dists  = np.linalg.norm(M_norm - pt, axis=1)
            nn_idx = np.argsort(dists)[:k_neighbors]
            w      = 1.0 / (dists[nn_idx] + 1e-8); w /= w.sum()

            neighbor_A = [A[i] for i in nn_idx]
            max_ts = max(len(a) for a in neighbor_A)
            max_th = max(len(ts) for a in neighbor_A for ts in a)

            result = []
            for ti in range(max_ts):
                ts_out = []
                for th in range(max_th):
                    sv = np.zeros(n_str)
                    for a, wi in zip(neighbor_A, w):
                        if ti < len(a) and th < len(a[ti]):
                            sv += wi * np.array(a[ti][th])
                    s = sv.sum()
                    if s > 1.0: sv /= s
                    ts_out.append(sv.tolist())
                result.append(ts_out)
            hull_pool.append(result)

        n_empty = sum(1 for c in inside_centres
                      if cell_counts.get(tuple(cell_of(c[None])[0].tolist()), 0) == 0)
        print(f"  [hull_fps_hybrid] dims={dims}: {len(inside_centres)} cells "
              f"({n_empty} empty) → +{len(hull_pool) - before} candidates")

    N = len(hull_pool)
    fps_n = max(1, int(N // hull_fps_ratio))
    print(f"  [hull_fps_hybrid] hull pool N={N}, fps pool target={fps_n} (ratio {hull_fps_ratio}:1)")

    # ── 2. Build fps novelty pool via RF + greedy FPS ─────────────────────────
    import time as _time
    from joblib import Parallel, delayed as _delayed
    fps_pool = []
    if N > 0:
        t0 = _time.perf_counter()
        mins, maxs = M_key.min(0), M_key.max(0)
        rng_m = np.where(maxs - mins > 0, maxs - mins, 1.0)
        M_key_norm = (M_key - mins) / rng_m

        A_flat   = np.array([_flatten_action_for_knn(a) for a in A])
        scaler_a = StandardScaler().fit(A_flat)
        scaler_m = StandardScaler().fit(M_key)
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=0)
        rf.fit(scaler_a.transform(A_flat), scaler_m.transform(M_key))
        print(f"  [fps] RF fit: {_time.perf_counter()-t0:.2f}s  (N={len(A)}, n_est={n_estimators})")

        # Generate fps_oversample * fps_n candidates: mix of mutations + random
        t1 = _time.perf_counter()
        n_mutation = fps_n * fps_oversample
        mutation_candidates = propose_candidates_by_pool_mutation(A, n_candidates=n_mutation // 2)
        random_candidates = propose_candidates_by_random(n_candidates=n_mutation // 2)
        candidates = mutation_candidates + random_candidates
        print(f"  [fps] candidate gen: {_time.perf_counter()-t1:.2f}s  ({len(candidates)} candidates)")

        # Flatten and predict
        t2 = _time.perf_counter()
        pool_flat = np.array([_flatten_action_for_knn(a) for a in candidates])
        pool_scaled = scaler_a.transform(pool_flat)
        pred_scaled = rf.predict(pool_scaled)
        pred_metrics = scaler_m.inverse_transform(pred_scaled)
        pred_norm = np.clip((pred_metrics - mins) / rng_m, 0.0, 1.0)
        print(f"  [fps] RF predict: {_time.perf_counter()-t2:.2f}s")

        # --- Rarity: KDE ---
        t3 = _time.perf_counter()
        try:
            kde = gaussian_kde(M_key_norm.T)
            density = kde.evaluate(pred_norm.T)
            log_rarity = -np.log(density + 1e-12)
        except Exception:
            log_rarity = np.ones(len(pred_norm)) * np.mean(np.var(M_key_norm, axis=0))
        print(f"  [fps] KDE rarity: {_time.perf_counter()-t3:.2f}s")

        # --- Uncertainty: vectorized batch covariance from RF trees ---
        t4 = _time.perf_counter()
        # Parallel per-tree predictions → (n_pool, n_trees, n_outputs)
        tree_preds = np.stack(
            Parallel(n_jobs=-1, prefer="threads")(
                _delayed(lambda t: t.predict(pool_scaled))(tree)
                for tree in rf.estimators_
            ), axis=1)
        n_pool = pool_scaled.shape[0]
        scale_vec = scaler_m.scale_
        print(f"  [fps] tree preds: {_time.perf_counter()-t4:.2f}s  ({n_pool} pts × {len(rf.estimators_)} trees)")

        t5 = _time.perf_counter()
        n_trees = tree_preds.shape[1]
        mean_pred = tree_preds.mean(axis=1, keepdims=True)          # (n_pool, 1, n_out)
        centered  = tree_preds - mean_pred                          # (n_pool, n_trees, n_out)
        cov_scaled_batch = np.einsum('bti,btj->bij', centered, centered) / max(n_trees - 1, 1)
        sv_outer = np.outer(scale_vec, scale_vec)                   # (n_out, n_out)
        cov_unscaled_batch = cov_scaled_batch * sv_outer[None, :, :]
        n_out = scale_vec.shape[0]
        eye   = np.eye(n_out)
        sign_b, logdet_b = np.linalg.slogdet(eye[None] + cov_unscaled_batch)
        trace_b = np.trace(cov_unscaled_batch, axis1=1, axis2=2)
        uncert_scores = np.where(sign_b > 0, logdet_b, trace_b)
        print(f"  [fps] uncertainty (vectorized): {_time.perf_counter()-t5:.2f}s")

        # --- Novelty ---
        lam = 0.1
        novelty = log_rarity + lam * uncert_scores

        # --- FPS: incremental min-distance update (O(fps_n × n_pool) instead of O(fps_n² × n_pool)) ---
        t6 = _time.perf_counter()
        # dist from each candidate to nearest existing training point (chunked to avoid OOM)
        chunk = 512
        dist_to_existing = np.full(n_pool, np.inf)
        for s in range(0, n_pool, chunk):
            e = min(s + chunk, n_pool)
            d = np.linalg.norm(pred_norm[s:e, None, :] - M_key_norm[None, :, :], axis=-1)
            dist_to_existing[s:e] = d.min(axis=1)
        print(f"  [fps] dist_to_existing: {_time.perf_counter()-t6:.2f}s")

        novelty_norm = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-12)
        dist_to_existing = dist_to_existing * (1.0 + novelty_norm)

        t7 = _time.perf_counter()
        # Incremental FPS: maintain running d_to_sel, update only with last selected point
        first = int(np.argmax(dist_to_existing))
        selected = [first]
        d_to_sel = np.linalg.norm(pred_norm - pred_norm[first], axis=1)  # dist to 1st selected
        combined_d = np.minimum(d_to_sel, dist_to_existing)
        for k in range(fps_n - 1):
            combined_d[selected] = -np.inf
            nxt = int(np.argmax(combined_d))
            selected.append(nxt)
            # Incremental update: only compute dist to the newly added point
            d_new = np.linalg.norm(pred_norm - pred_norm[nxt], axis=1)
            d_to_sel  = np.minimum(d_to_sel, d_new)
            combined_d = np.minimum(d_to_sel, dist_to_existing)
            if k % 200 == 0:
                print(f"  [fps] FPS iter {k}/{fps_n-1}  ({_time.perf_counter()-t7:.1f}s elapsed)")
        print(f"  [fps] FPS total: {_time.perf_counter()-t7:.2f}s  ({fps_n} selected from {n_pool})")

        fps_pool = [candidates[i] for i in selected]
        print(f"  [hull_fps_hybrid] fps novelty pool size={len(fps_pool)} "
              f"(from {len(candidates)} candidates)  total propose: {_time.perf_counter()-t0:.1f}s")

    # ── 3. Merge and randomly sample ─────────────────────────────────────────
    combined = hull_pool + fps_pool
    print(f"  [hull_fps_hybrid] combined pool={len(combined)} → random sample {n_candidates}")

    if not combined:
        print("  [hull_fps_hybrid] empty pool, falling back to random")
        return propose_candidates_by_random(n_candidates=n_candidates)
    if len(combined) <= n_candidates:
        return combined

    idx = np.random.choice(len(combined), size=n_candidates, replace=False)
    return [combined[i] for i in idx]


def initial_candidates(bounds, n_candidates, num_max_threads=20):
    """ Generate initial candidates as one-hot vectors within the given bounds. """
    k = len(bounds)
    candidates = []
    zero_action_weights = [0.0] * k
    for i in range(n_candidates):
        initial_action_weights = [0.0] * k
        for j in range(k):
            if i == j:
                initial_action_weights[j] = 1.0

        candidate_max_threads = num_max_threads
        if i == 10 or i == 11:
            # heuristic for I/O bound actions
            candidate_max_threads = 1
        elif i == 12:
            candidate_max_threads = 4
        for num_threads in range(1, candidate_max_threads + 1):
            candidate = [initial_action_weights for _ in range(num_threads)]
            while len(candidate) < num_max_threads:
                candidate.append(zero_action_weights)

            for weight in [0.2, 0.4, 0.6, 0.8, 1.0]:
                scaled_candidate = [[w * weight for w in thread] for thread in candidate]
                if i < 10:
                    random.shuffle(scaled_candidate)
                    candidates.append(scaled_candidate)
                else:
                    for _ in range(num_max_threads - num_threads + 1):
                        random.shuffle(scaled_candidate)
                        candidates.append(scaled_candidate)

    final_candidates = []
    weight = 1.0
    for action in candidates:
        scaled_action = [[w * weight for w in thread] for thread in action]
        final_candidates.append([scaled_action, scaled_action])

    return final_candidates


def write_to_hdf5(action_weights, file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('execution_plan', data=action_weights)


def write_actions_to_execution_plans(actions, destination_path: str, profiling_machines: list[str]):
    machines = [Machine.from_hostname(hostname) for hostname in profiling_machines]
    num_machines = len(machines)
    chunk_size = len(actions) // num_machines  # Ceiling division
    extra_data_points = len(actions) % num_machines

    for chunk_idx in range(len(machines)):
        os.makedirs(f"{destination_path}/chunk_{chunk_idx}/plans", exist_ok=True)
        if chunk_idx < extra_data_points:
            action_chunk = actions[chunk_idx * (chunk_size + 1):(chunk_idx + 1) * (chunk_size + 1)]
        else:
            action_chunk = actions[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        for action_idx, action in enumerate(action_chunk):
            file_path = f"{destination_path}/chunk_{chunk_idx}/plans/plan_{action_idx:06d}.h5"
            updated_action = []
            for action_timestamp in action:
                if all(all(value == 0.0 for value in thread) for thread in action_timestamp):
                    continue
                updated_action.append(action_timestamp)

            write_to_hdf5(updated_action, file_path)


def filter_high_variance_data(A, M, var, metric_ranges, threshold=0.1):
    metric_ranges = metric_ranges[NUM_METRICS * 1:NUM_METRICS * 2]

    # Calculate min and max for each metric
    if len(M) > 0:
        # Compute dynamic thresholds for each metric
        dynamic_thresholds = metric_ranges * threshold

        filtered_A = []
        filtered_M = []
        filtered_var = []

        for a, m, v in zip(A, M, var):
            # Check if variance is below threshold for each metric dimension
            v_target = v[NUM_METRICS * 1:NUM_METRICS * 2]
            if np.all(v_target <= dynamic_thresholds):
                filtered_A.append(a)
                filtered_M.append(m)
                filtered_var.append(v)
            else:
                print(f"Filtering out sample with high variance: {v_target} > {dynamic_thresholds}")

        # Print how many samples were filtered out
        orig_size = len(A)
        filtered_size = len(filtered_A)
        print(f"Filtered {orig_size - filtered_size} samples with high variance ({(orig_size - filtered_size) / orig_size * 100:.2f}%)")
        print(f"Dynamic thresholds: {dynamic_thresholds}")
        return filtered_A, np.array(filtered_M), np.array(filtered_var)
    else:
        return A, M, var

# ---------------------------------------------------------------------------
# hybrid proposer (rounds 1+)
# ---------------------------------------------------------------------------

def propose_convex_hull_and_novelty_mix(A, M, n_candidates):
    """via hull_mixed_fps_hybrid"""
    n_hull_fps = n_candidates

    print(f"  [propose] n_hull_fps={n_hull_fps} (hull_fps_ratio={HULL_FPS_RATIO:.3f})")

    hull_fps = propose_by_hull_mixed_fps_hybrid(
        A, M, n_hull_fps, hull_fps_ratio=HULL_FPS_RATIO)

    random.shuffle(hull_fps)
    print(f"  [propose] total={len(hull_fps)}")
    return hull_fps


# ---------------------------------------------------------------------------
# Build verification
# ---------------------------------------------------------------------------

def verify_build(machines, user_name, private_key_path):
    """
    SSH into each machine in parallel and run a quick bazel build to confirm
    the benchmark compiles cleanly.  Prints a per-machine pass/fail summary
    and returns True only if every machine succeeds.
    """
    import concurrent.futures

    build_cmd = (
        "cd ~/fleetbench && "
        "bazel build --config=clang --config=opt "
        "fleetbench/mimesys:mimesys_benchmark 2>&1 | tail -3 && "
        "echo BUILD_OK"
    )

    def _check_one(hostname):
        machine = Machine.from_hostname(hostname)
        try:
            client, sftp = machine.initialize_connection(user_name, private_key_path)
            _, stdout, stderr = client.exec_command(build_cmd)
            exit_status = stdout.channel.recv_exit_status()
            out = stdout.read().decode().strip()
            machine.close_connection(sftp, client)
            if exit_status == 0 and "BUILD_OK" in out:
                print(f"  [OK]   {hostname}  — {out.splitlines()[-1] if out else 'build OK'}")
                return hostname, True
            else:
                err = stderr.read().decode().strip()
                print(f"  [FAIL] {hostname}  — exit={exit_status}  {err[-200:]}")
                return hostname, False
        except Exception as exc:
            print(f"  [ERR]  {hostname}  — {exc}")
            return hostname, False

    print(f"\n=== Build verification ({len(machines)} machines, parallel) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(machines)) as executor:
        futures = [executor.submit(_check_one, h) for h in machines]
        results = dict(f.result() for f in concurrent.futures.as_completed(futures))

    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    print(f"\nBuild check: {n_ok}/{len(machines)} passed, {n_fail} failed")
    if n_fail:
        failed = [h for h, ok in results.items() if not ok]
        print(f"  Failed machines: {failed}")
        print("  Fix the build on failed machines before running collection.")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Round loading helper
# ---------------------------------------------------------------------------

def load_round(profiler, round_idx):
    """Load plan_stat_pairs for an already-collected round_{round_idx}."""
    dest = os.path.join(OUTPUT_PATH, f"round_{round_idx}")
    found_files = {}
    for root, _, files in os.walk(dest):
        for fname in files:
            m = re.search(r"validation-(\d+)\.zip$", fname)
            if m:
                found_files[int(m.group(1))] = fname
    if not found_files:
        return []
    print(f"  loading round_{round_idx}: {len(found_files)} chunks from disk")
    return profiler.parse_metrics_from_zip(found_files, dest)


def pairs_to_AM(pairs, max_len=0):
    """Convert plan_stat_pairs → (A list, M array, var array) with padding."""
    valid = [p for p in pairs if p is not None]
    if not valid:
        return [], None, None
    cur_max = max(len(p[2]) for p in valid)
    max_len = max(max_len, cur_max)
    A   = [p[0] for p in valid]
    M   = np.array([p[2] + [0.0]*(max_len - len(p[2])) for p in valid], dtype=float)
    var = np.array([p[3] + [0.0]*(max_len - len(p[3])) for p in valid], dtype=float)
    return A, M, var


# ---------------------------------------------------------------------------
# Metric summary
# ---------------------------------------------------------------------------

def report_metrics(M, label=""):
    labels = ["CPU%", "IO KB/s", "L3-S0", "L3-S1", "BW-S0", "BW-S1"]
    M_key  = M[:, _KEY_METRIC_INDICES]
    hdr = f"  [{label}]" if label else ""
    print(f"\n{'Metric':<12}  {'min':>10}  {'median':>10}  {'max':>10}  (N={len(M)}){hdr}")
    print("-" * 56)
    for lbl, col in zip(labels, M_key.T):
        print(f"{lbl:<12}  {col.min():>10.2f}  {np.median(col):>10.2f}  {col.max():>10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_collection(n_rounds: int, restart: bool = False):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    profiler = Profiler(InitializeRequest(
        user_name=SSH_USER,
        private_key_path=SSH_KEY_PATH,
        worker_host_names=PROFILING_MACHINES,
        my_hostname=MY_HOSTNAME,
    ))

    # ── Build verification (skipped) ─────────────────────────────────────────
    # build_ok = verify_build(PROFILING_MACHINES, SSH_USER, SSH_KEY_PATH)
    # if not build_ok:
    #     raise SystemExit("Aborting: one or more machines failed the build check.")

    # ── Optional restart: wipe existing rounds ────────────────────────────────
    if restart:
        existing_dirs = [
            d for d in os.listdir(OUTPUT_PATH)
            if re.match(r"round_\d+$", d) and os.path.isdir(os.path.join(OUTPUT_PATH, d))
        ]
        if existing_dirs:
            print(f"\n--restart: removing {len(existing_dirs)} existing round dirs "
                  f"from {OUTPUT_PATH}")
            for d in existing_dirs:
                shutil.rmtree(os.path.join(OUTPUT_PATH, d))

    # ── Detect already-completed rounds ──────────────────────────────────────
    existing = sorted(
        int(re.search(r"\d+", d).group())
        for d in os.listdir(OUTPUT_PATH)
        if re.match(r"round_\d+$", d) and os.path.isdir(os.path.join(OUTPUT_PATH, d))
    )
    print(f"Existing rounds: {existing}")

    # ── Round 0: initial_candidates ──────────────────────────────────────────
    if 0 not in existing:
        bounds = [(0.0, 1.0)] * NUM_ACTIONS
        A_init = initial_candidates(bounds, NUM_ACTIONS, NUM_THREADS)
        print(f"\n=== Round 0  (initial_candidates: {len(A_init)} plans) ===")
        dest0 = os.path.join(OUTPUT_PATH, "round_0")
        write_actions_to_execution_plans(A_init, dest0, PROFILING_MACHINES)
        pairs0 = asyncio.run(profiler.profile_actions(dest0))
        n_valid0 = sum(1 for p in pairs0 if p is not None)
        print(f"  Valid: {n_valid0}/{len(pairs0)}")
        existing.append(0)
    else:
        print("  round_0 already on disk, loading …")
        pairs0 = load_round(profiler, 0)
        n_valid0 = sum(1 for p in pairs0 if p is not None)
        print(f"  round_0: {n_valid0} valid samples")

    A0, M0, var0 = pairs_to_AM(pairs0)
    if M0 is None:
        print("ERROR: round_0 has no valid data. Aborting.")
        return

    # Filter variance in round 0
    mins = np.min(M0, axis=0); maxs = np.max(M0, axis=0)
    A0, M0, var0 = filter_high_variance_data(A0, M0, var0, maxs - mins, threshold=0.1)
    print(f"  After variance filter: {len(A0)} samples")
    report_metrics(M0, "round 0")

    # ── Load any existing rounds 1+ ───────────────────────────────────────────
    A, M, var = list(A0), M0, var0
    for r_idx in sorted(r for r in existing if r > 0):
        pairs = load_round(profiler, r_idx)
        A_r, M_r, var_r = pairs_to_AM(pairs, max_len=M.shape[1])
        if M_r is None:
            continue
        # Pad width if needed
        pad = M_r.shape[1] - M.shape[1]
        if pad > 0:
            M   = np.pad(M,   ((0,0),(0,pad)))
            var = np.pad(var, ((0,0),(0,pad)))
        A.extend(A_r)
        M   = np.vstack([M, M_r])
        var = np.vstack([var, var_r])

    start_round = (max(existing) + 1) if existing else 1
    print(f"\nStarting active-learning from round {start_round} "
          f"(dataset: {len(A)} samples)")

    # ── Active-learning rounds 1..n_rounds ────────────────────────────────────
    for r in range(start_round, n_rounds + 1):
        print(f"\n=== Round {r}  (batch={BATCH_SIZE}, strategy=hull:fps 5:5) ===")

        batch = propose_convex_hull_and_novelty_mix(A, M, BATCH_SIZE)

        dest = os.path.join(OUTPUT_PATH, f"round_{r}")
        write_actions_to_execution_plans(batch, dest, PROFILING_MACHINES)

        pairs = asyncio.run(profiler.profile_actions(dest))
        A_r, M_r, var_r = pairs_to_AM(pairs, max_len=M.shape[1])
        n_valid = 0 if M_r is None else len(A_r)
        print(f"  Valid: {n_valid}/{len(pairs)}")

        if M_r is None:
            print("  No valid results, skipping round")
            continue

        # Pad width
        pad = M_r.shape[1] - M.shape[1]
        if pad > 0:
            M   = np.pad(M,   ((0,0),(0,pad)))
            var = np.pad(var, ((0,0),(0,pad)))

        mins_new = np.min(np.vstack([M, M_r]), axis=0)
        maxs_new = np.max(np.vstack([M, M_r]), axis=0)
        A_r, M_r, var_r = filter_high_variance_data(
            A_r, M_r, var_r, maxs_new - mins_new, threshold=0.1)

        A.extend(A_r)
        M   = np.vstack([M, M_r])
        var = np.vstack([var, var_r])

        M_key = M[:, _KEY_METRIC_INDICES]
        print(f"  Cumulative: {len(A)} samples | "
              f"IO med={np.median(M_key[:,1]):.0f}  CPU med={np.median(M_key[:,0]):.1f}%")

    report_metrics(M, f"final — round 0 + {n_rounds} active rounds")
    print(f"\nDone. {len(A)} samples → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of active-learning rounds after round 0 (default: 50)")
    parser.add_argument("--restart", action="store_true",
                        help="Delete all existing round dirs and start fresh from round 0")
    args = parser.parse_args()

    print(f"collect_surrogate_v2: round 0 (initial_candidates) + {args.rounds} "
          f"active rounds → {OUTPUT_PATH}"
          + (" [RESTART]" if args.restart else ""))
    run_collection(n_rounds=args.rounds, restart=args.restart)
