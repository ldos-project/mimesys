"""
collect_training_data.py
=======================
Active-learning data collection with 13-action space
(10 non-IO + Readahead[10] + Fallocate_4MB[11] + Hdd_1MB[12]).

Round 0  : initial_candidates  — one-hot style sweep covering each action
           in isolation at various thread counts and weight scales (~1300 plans).
Rounds 1+: hull:fps = 5:5 (io mutation disabled)
           - 50 % from hull interpolation  }
           - 50 % from fps novelty          } via propose_by_hull_mixed_fps_hybrid (hull_fps_ratio=1)

Output:    ~/mimesys_training_data/training_data_v1

Usage:
  cd mimesys/collection/scripts
  python collect_training_data.py [--rounds 50] [--restart]
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

# Pull SSH credentials, worker host list, and controller hostname from the
# same worker_scripts/config.py that install_remote_dependencies.py uses, so
# there is one source of truth for "which hosts is this controller talking to".
_WORKER_SCRIPTS_DIR = os.path.join(REPO_ROOT, "worker_scripts")
if _WORKER_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _WORKER_SCRIPTS_DIR)
import config as worker_config

OUTPUT_PATH = os.path.expanduser(
    os.environ.get("MIMESYS_OUTPUT_PATH",
                    "~/mimesys_training_data/training_data_v2_2sec"))

PROFILING_MACHINES = list(worker_config.HOSTNAMES)
SSH_USER           = worker_config.USERNAME
SSH_KEY_PATH       = os.path.expanduser(worker_config.PRIVATE_KEY_PATH)
MY_HOSTNAME        = worker_config.MY_HOSTNAME

NUM_ACTIONS    = 13
NUM_THREADS    = 20
BATCH_SIZE     = worker_config.PER_MACHINE_BATCH * len(PROFILING_MACHINES)   # active-learning round size
# 24 = 20 per-core CPU% + 1 avg_cpu_utilizations_total + 1 io + 1 l3_cache_usage + 1 memory_bandwidth
NUM_METRICS    = 24

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

        # Per-thread uniform[0, 1] scale on curr_action so per-core CPU%
        # spreads across the full [0, 100] range. Without this, mutations
        # inherit parent magnitudes (which sum ~1 per thread → saturated cores),
        # producing the bimodal per-core distribution.
        curr_idx = len(result_arrs) - 1
        curr = result_arrs[curr_idx]
        thread_scales = np.random.uniform(0.8, 1.0, size=curr.shape[0])
        result_arrs[curr_idx] = curr * thread_scales[:, None]

        candidates.append(_arrays_to_action(result_arrs))

    return candidates


# M layout: [prev_metrics(NUM_METRICS), curr_metrics(NUM_METRICS), noop_metrics(NUM_METRICS)] = NUM_METRICS * 3 dims total.
# M ordering verified: group 0 = prev_action, group 1 = curr_action, group 2 = no-op.
# Within each 24-dim group the last 4 are the key metrics in this order:
#   idx+20  avg_cpu_utilizations_total
#   idx+21  io
#   idx+22  l3_cache_usage    (aggregate; per-socket dropped to avoid CHA artifact)
#   idx+23  memory_bandwidth  (aggregate)
# For curr_action (group 1, offset 24): flat indices 44–47.
_KEY_METRIC_INDICES = [
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


AUG_FACTOR = 10  # variants per original (matches dataloader's training-time aug_factor)


def _augment_pool_for_hull(A, M, aug_factor: int = AUG_FACTOR):
    """Expand the (A, M) pool by `aug_factor` so hull/FPS don't treat thread
    permutations or socket swaps as novel.

    For each (a, m) we emit `aug_factor` variants:
      0. identity
      1..half-1                : intra-socket thread perm (random per variant)
      half                     : socket swap (no perm)
      half+1..end              : intra-socket perm + socket swap

    Notes:
      - Intra-perm preserves all socket-aggregate features hull/FPS use
        (cpu_s0/s1 aggregate, cpu_std, cpu_mid_frac are invariant) but adds
        action-space diversity so FPS mutation has more parent shapes.
      - Socket swap exchanges cpu_s0↔cpu_s1 (per-core indices) only. LLC and
        memory bandwidth are now socket-aggregated so they're already
        swap-invariant.
      - Mirrors `augment_dataset` in preprocessing/dataloader.py so training-time
        aug doesn't grant the model "free" samples that collection already paid
        for.
    """
    def _action_apply(a, perm_s0, perm_s1, do_swap):
        out = []
        for arr in a:
            arr_np = np.asarray(arr, dtype=np.float32)
            new = arr_np.copy()
            if perm_s0 is not None:
                new[0:10]  = arr_np[0:10][perm_s0]
                new[10:20] = arr_np[10:20][perm_s1]
            if do_swap:
                new = np.vstack([new[10:20], new[0:10]])
            out.append(new.tolist() if isinstance(arr, list) else new)
        return out

    def _metric_apply(m, perm_s0, perm_s1, do_swap):
        new = np.asarray(m, dtype=float).copy()
        for K in range(3):                              # prev, curr, noop
            base = K * NUM_METRICS                      # 0, 24, 48
            if perm_s0 is not None:
                new[base:base+10]    = np.asarray(m[base:base+10])[perm_s0]
                new[base+10:base+20] = np.asarray(m[base+10:base+20])[perm_s1]
            if do_swap:
                tmp_s0 = new[base:base+10].copy()
                new[base:base+10]    = new[base+10:base+20]
                new[base+10:base+20] = tmp_s0
                # LLC/BW are now aggregated → no per-socket swap needed
        return new

    n_extra = aug_factor - 1
    half = n_extra // 2
    A_aug, M_aug = [], []
    for a, m in zip(A, M):
        # 0: identity
        A_aug.append(a); M_aug.append(np.asarray(m, dtype=float))
        # 1..n_extra: aug variants — perm only (first half) vs perm+swap (second half)
        # Variant `half` is socket-swap only (no perm).
        for k in range(n_extra):
            do_perm = (k != half)
            do_swap = (k >= half)
            ps0 = np.random.permutation(10) if do_perm else None
            ps1 = np.random.permutation(10) if do_perm else None
            A_aug.append(_action_apply(a, ps0, ps1, do_swap))
            M_aug.append(_metric_apply(m, ps0, ps1, do_swap))
    return A_aug, np.asarray(M_aug)


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
    Metric indices (new 23-D layout with socket-aggregated LLC/BW):
                    0=CPU%  1=IO  2=LLC  3=BW  (key metrics from curr_action group)
                    4=CPU-S0  5=CPU-S1  (per-socket CPU util, cols 24-33 / 34-43)
                    6=CPU-STD  7=CPU-MID-FRAC  (per-core shape — std across 20 cores
                    and fraction of cores at 20-80% mid-range)
    """
    from collections import Counter

    # Augment pool with thread-perm + socket-swap variants so hull/FPS don't
    # treat augmentation-equivalents of existing samples as novel candidates.
    n_orig = len(A)
    A, M = _augment_pool_for_hull(A, M)
    print(f"  [hull/fps] augmented pool: {n_orig} → {len(A)} samples "
          f"({AUG_FACTOR}× via intra-perm + socket-swap variants)")

    M_key4 = M[:, _KEY_METRIC_INDICES].astype(float)
    # Per-core CPU% live in group 1 (curr_action), cols 24..43 (alphabetical
    # core_00..core_19). Socket 0 = cores 0..9, socket 1 = cores 10..19.
    G1_CPU_LO = NUM_METRICS               # 24
    G1_CPU_HI = NUM_METRICS + 20          # 44
    per_core_curr = M[:, G1_CPU_LO:G1_CPU_HI].astype(float)
    cpu_s0 = per_core_curr[:, :10].mean(axis=1, keepdims=True)
    cpu_s1 = per_core_curr[:, 10:].mean(axis=1, keepdims=True)
    # Per-core shape features: encourage workloads that put cores in the
    # mid-range (20..80%) rather than the bimodal 0/100 tails.
    cpu_std      = per_core_curr.std(axis=1, keepdims=True)
    cpu_mid_frac = ((per_core_curr >= 20.0) & (per_core_curr <= 80.0)) \
                     .mean(axis=1, keepdims=True) * 100.0
    M_key = np.hstack([M_key4, cpu_s0, cpu_s1, cpu_std, cpu_mid_frac])  # (N, 8)

    # ── 1. Build hull pool ────────────────────────────────────────────────────
    # Index legend: 0=CPU%, 1=IO, 2=LLC, 3=BW, 4=CPU-S0, 5=CPU-S1,
    #               6=CPU-STD, 7=CPU-MID-FRAC.
    GROUPS = [
        (0, 2, 3),    # CPU% × LLC × BW   (compute-heavy hull)
        (1, 2, 3),    # IO   × LLC × BW   (io-heavy hull)
        (0, 1, 2),    # CPU% × IO  × LLC
        (0, 1, 3),    # CPU% × IO  × BW
        (0, 1),       # CPU% × IO
        (2, 3),       # LLC  × BW         (cache vs memory traffic)
        (4, 5),       # cpu-s0 vs cpu-s1  (socket CPU asymmetry)
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
        if i >= 10:
            # heuristic for I/O bound actions
            candidate_max_threads = 4
        # elif i == 12:
        #     candidate_max_threads = 4
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

def _action_purity_per_core(action_curr, kernel_thresh=0.5):
    """Return list of (core_idx, dom_kernel) for each core where the dominant
    kernel has weight >= kernel_thresh (i.e., the core is 'pure' on that kernel).
    `action_curr` is the per-thread×per-kernel matrix at curr-timestep
    (list[NUM_THREADS][NUM_ACTIONS]).
    """
    out = []
    for c, row in enumerate(action_curr):
        if not row: continue
        s = sum(row)
        if s < 0.05: continue
        max_w = max(row)
        if max_w >= kernel_thresh:
            out.append((c, int(np.argmax(row))))
    return out


def propose_disjoint_composites(A, n_candidates, num_actions=NUM_ACTIONS,
                                 num_threads=NUM_THREADS, kernel_thresh=0.5,
                                 fallback_synthetic_prob=0.5):
    """Pair two 'pure' parents from pool A with **disjoint active cores** and
    union them: each core takes weights from whichever parent had it active.

    A 'pure' parent here is an action whose active cores are dominated by a
    single kernel each. Goal: produce composite actions of the form
        kernel K1 on cores C1  +  kernel K2 on cores C2  (C1 ∩ C2 = ∅)
    where K1 ≠ K2 — directly targeting the 'IO-CPU additivity' gap
    (no such composites exist in the v6 corpus).

    Falls back to a synthetic pure parent (one kernel on a random core subset
    with weight in [0.5, 1.0]) when pool A doesn't have enough pure samples
    for the desired kernel pair (controlled by `fallback_synthetic_prob`).

    Action layout: list of timestamps, each = list[num_threads][num_actions].
    We construct two timestamps (prev_action + curr_action) so the format
    matches the rest of the pipeline.
    """
    if not A:
        # No pool yet — produce all synthetic
        purified = []
    else:
        # Identify pure parents in A — index by dominant-kernel sets
        purified = []   # list of (action_arr, dict{core: dom_kernel})
        for a in A:
            curr_ts_idx = 1 if len(a) > 1 else 0
            curr = a[curr_ts_idx]
            pcores = _action_purity_per_core(curr, kernel_thresh)
            if not pcores: continue
            # 'Mostly pure': at least 80% of active cores are dominated by one kernel
            active = sum(1 for row in curr if row and sum(row) > 0.05)
            if len(pcores) >= max(1, int(0.8 * active)):
                # Group cores by dominant kernel for this parent
                by_kernel = {}
                for c, k in pcores:
                    by_kernel.setdefault(k, []).append(c)
                # Single-kernel parents only — pick the dominant one if multiple
                dom_kernel = max(by_kernel, key=lambda k: len(by_kernel[k]))
                cores      = by_kernel[dom_kernel]
                purified.append((np.asarray(curr, dtype=float), dom_kernel, set(cores)))

    print(f"  [composite] pool A had {len(A)} actions; {len(purified)} pure parents available")
    pure_by_kernel = {}
    for arr, k, cs in purified:
        pure_by_kernel.setdefault(k, []).append((arr, cs))

    def _make_synthetic_pure(kernel, n_cores_range=(2, 10), w_range=(0.5, 1.0)):
        """Build a synthetic pure parent: `kernel` on a random subset of cores
        with random weight in [w_range], others idle."""
        n_cores = random.randint(*n_cores_range)
        # Pick random core subset
        cores = random.sample(range(num_threads), n_cores)
        w = random.uniform(*w_range)
        arr = np.zeros((num_threads, num_actions), dtype=float)
        for c in cores:
            arr[c, kernel] = w
        return arr, set(cores)

    # Kernel groups so we can bias toward cross-resource composites
    CPU_HEAVY = [5, 7, 8, 9]            # SWISSMAP_InsertManyOrdered/InsertMiss
    IO_PURE   = [10, 11, 12]            # STRESS_NG_* (IO-bound)
    MID_KERN  = [2, 3, 4, 6]            # LIBC/SIMD/SWISSMAP-mid
    HASH_K    = [0, 1]                  # HASHING

    candidates = []
    for _ in range(n_candidates):
        # Decide pair-of-kernel categories — bias toward CPU_HEAVY × IO_PURE
        # to directly close the (k=8 + k=12) gap; sample others too for breadth.
        roll = random.random()
        if roll < 0.50:
            grp1, grp2 = CPU_HEAVY, IO_PURE
        elif roll < 0.75:
            grp1, grp2 = CPU_HEAVY, MID_KERN
        elif roll < 0.90:
            grp1, grp2 = MID_KERN, IO_PURE
        else:
            grp1, grp2 = HASH_K, IO_PURE

        k1 = random.choice(grp1); k2 = random.choice(grp2)
        if k1 == k2: continue  # rare overlap

        # Pull pure parents (from pool A or synthetic)
        def _get_parent(kernel):
            real = pure_by_kernel.get(kernel, [])
            if real and random.random() > fallback_synthetic_prob:
                return random.choice(real)
            return _make_synthetic_pure(kernel)

        p1_arr, p1_cores = _get_parent(k1)
        p2_arr, p2_cores = _get_parent(k2)

        # Disjoint: if overlap, reassign p2 cores to the complement
        overlap = p1_cores & p2_cores
        if overlap:
            free = list(set(range(num_threads)) - p1_cores)
            if len(free) < len(p2_cores):
                # Truncate p2 to fit the free pool
                p2_cores = set(random.sample(free, min(len(free), len(p2_cores))))
            else:
                p2_cores = set(random.sample(free, len(p2_cores)))
            # Rebuild p2_arr to live only on the new cores
            new_p2 = np.zeros_like(p2_arr)
            for c in p2_cores:
                new_p2[c, k2] = random.uniform(0.5, 1.0)
            p2_arr = new_p2

        # Union: prefer p1 weights on its cores, p2 on its cores; idle elsewhere
        composite = np.zeros((num_threads, num_actions), dtype=float)
        for c in p1_cores:
            composite[c] = p1_arr[c]
        for c in p2_cores:
            composite[c] = p2_arr[c]

        # Apply a per-thread uniform[0.8, 1.0] scale so intensity varies
        thread_scales = np.random.uniform(0.8, 1.0, size=num_threads)
        composite = composite * thread_scales[:, None]

        # Build two-timestep action (prev = composite, curr = composite for now;
        # active-learning loop uses index 1 as curr_action)
        action = [composite.tolist(), composite.tolist()]
        candidates.append(action)

    return candidates


def _action_key(action, decimals=2):
    """Canonical dedup key — hash of rounded curr_action weights.

    Rounds to 2 decimal places (matches the empirical observation that ~99%
    of v6 duplicates are byte-identical, with no FP-noise variants). Hashing
    bytes is O(n_threads * n_kernels) and far cheaper than equality scans.
    """
    curr_idx = 1 if len(action) > 1 else 0
    return hash(np.round(np.asarray(action[curr_idx], dtype=float), decimals).tobytes())


def _restrict_to_single_socket(actions, mode="alternate", active_thresh=1e-6):
    """Repack each action so all active threads land in ONE socket's core slots.
    On c220g5: socket 0 = thread slots 0-9, socket 1 = thread slots 10-19.

    mode='alternate' (default): alternate s0/s1 assignment per action (balanced).
    mode='s0' | 's1': force all actions to that socket.
    mode='random': uniform random per action.

    Thread identity is preserved across timestamps: a thread's row in prev/curr
    moves together. If an action has >10 active threads, the lowest-indexed 10
    are kept and the rest are dropped (single-socket holds 10 cores).
    """
    N_THREADS = 20
    N_PER_SOCKET = 10
    for i, a in enumerate(actions):
        if not a or not a[0]:
            continue
        if mode == "alternate":
            target = i % 2
        elif mode == "s0":
            target = 0
        elif mode == "s1":
            target = 1
        else:
            target = random.randint(0, 1)
        base = 0 if target == 0 else 10
        n_kernels = len(a[0][0])
        active = []
        for j in range(len(a[0])):
            for ts in a:
                if j < len(ts) and any(w > active_thresh for w in ts[j]):
                    active.append(j); break
        active = active[:N_PER_SOCKET]
        new_action = []
        for ts in a:
            new_ts = [[0.0] * n_kernels for _ in range(N_THREADS)]
            for slot_i, orig_j in enumerate(active):
                if orig_j < len(ts):
                    new_ts[base + slot_i] = list(ts[orig_j])
            new_action.append(new_ts)
        actions[i] = new_action
    return actions


def _apply_action_noise(actions, eps=0.1, renormalize=True, active_thresh=1e-6):
    """Add Uniform(-eps, +eps) noise to ACTIVE cells only (weight > active_thresh).
    Idle cells (weight == 0) stay exactly 0 — preserving the idle-core structure.
    Clips perturbed cells to [0, 1]. If renormalize=True and a thread's weight
    sums to > 1, rescales that thread back down to sum=1. Mutates in place —
    also returns the list for chaining.
    """
    if eps <= 0: return actions
    for a in actions:
        for ts_idx in range(len(a)):
            arr = np.asarray(a[ts_idx], dtype=float)
            if arr.size == 0: continue
            active_mask = arr > active_thresh
            if not active_mask.any(): continue
            noise = np.random.uniform(-eps, eps, size=arr.shape)
            # Apply noise only on active cells
            arr[active_mask] = np.clip(arr[active_mask] + noise[active_mask], 0.0, 1.0)
            if renormalize:
                sums = arr.sum(axis=-1, keepdims=True)
                mask = (sums > 1.0).squeeze(-1) if sums.ndim > 1 else (sums > 1.0)
                if mask.any():
                    arr[mask] = arr[mask] / sums[mask]
            a[ts_idx] = arr.tolist()
    return actions


def propose_convex_hull_and_novelty_mix(A, M, n_candidates):
    """via hull_mixed_fps_hybrid, optionally mixed with disjoint composites.

    Env vars:
      MIMESYS_COMPOSITE_FRACTION (default 0.0)  fraction of batch from composites
      MIMESYS_DEDUP              (default "1")  set to "0" to disable dedup
    """
    composite_frac = float(os.environ.get("MIMESYS_COMPOSITE_FRACTION", "0.0"))
    composite_frac = max(0.0, min(1.0, composite_frac))
    n_composite = int(round(n_candidates * composite_frac))
    n_hull_fps  = n_candidates - n_composite
    do_dedup    = os.environ.get("MIMESYS_DEDUP", "1") != "0"

    # Build pool-A dedup set (existing actions we already collected)
    if do_dedup and A:
        pool_keys = {_action_key(a) for a in A}
        print(f"  [dedup] pool A: {len(A)} actions → {len(pool_keys)} unique signatures "
              f"({100*(len(A)-len(pool_keys))/max(len(A),1):.1f}% dupes already)")
    else:
        pool_keys = set()

    # Over-propose hull/FPS to absorb dedup drops (composites use random
    # uniform weights so collisions are already astronomically rare)
    OVERSHOOT = 1.4
    raw_hull = (propose_by_hull_mixed_fps_hybrid(
                    A, M, int(n_hull_fps * OVERSHOOT), hull_fps_ratio=HULL_FPS_RATIO)
                if n_hull_fps > 0 else [])
    raw_comp = (propose_disjoint_composites(A, n_composite)
                if n_composite > 0 else [])

    def _dedup_filter(cands, seen):
        """Drop entries whose key is in `seen` or pool_keys; update `seen` with kept keys."""
        kept = []
        for c in cands:
            k = _action_key(c)
            if k in seen or k in pool_keys: continue
            seen.add(k)
            kept.append(c)
        return kept

    used = set()
    if do_dedup:
        kept_hull = _dedup_filter(raw_hull, used)
        kept_comp = _dedup_filter(raw_comp, used)
    else:
        kept_hull = list(raw_hull)
        kept_comp = list(raw_comp)

    # Truncate to target
    hull_fps   = kept_hull[:n_hull_fps]
    composites = kept_comp[:n_composite]

    short = n_candidates - len(hull_fps) - len(composites)
    if short > 0:
        # Top-up with extra random proposals if dedup left us short
        extra = propose_candidates_by_random(n_candidates=int(short * 1.5))
        if do_dedup:
            extra = _dedup_filter(extra, used)
        composites += extra[:short]

    out = hull_fps + composites
    random.shuffle(out)
    dropped_hull = len(raw_hull) - len(hull_fps)
    dropped_comp = max(0, len(raw_comp) - len(composites))
    print(f"  [propose] n_hull_fps={n_hull_fps}  n_composite={n_composite} "
          f"(composite_frac={composite_frac:.2f}, dedup={do_dedup})")
    print(f"  [propose] kept hull/fps={len(hull_fps)} (dropped {dropped_hull}); "
          f"composite={len(composites)}; total={len(out)}")

    # Per-cell noise injection (default off; env var MIMESYS_ACTION_NOISE_EPS)
    noise_eps = float(os.environ.get("MIMESYS_ACTION_NOISE_EPS", "0.0"))
    if noise_eps > 0:
        out = _apply_action_noise(out, eps=noise_eps, renormalize=True)
        print(f"  [propose] applied per-cell noise eps=±{noise_eps:.2f} "
              f"(clip→[0,1], renorm if thread_sum>1)")

    # Single-socket restriction (default off; env MIMESYS_SINGLE_SOCKET={alternate,s0,s1,random})
    single_sock = os.environ.get("MIMESYS_SINGLE_SOCKET", "").strip()
    if single_sock:
        out = _restrict_to_single_socket(out, mode=single_sock)
        print(f"  [propose] restricted to single socket (mode={single_sock})")

    return out


# ---------------------------------------------------------------------------
# Round loading helper
# ---------------------------------------------------------------------------

def _ensure_extracted(round_dir):
    """Make sure each chunk_X/ has its validation-*.zip extracted into chunk_X/results/.
    The collection pipeline already extracts during profile_actions, but if this fn is
    called against an existing on-disk round (e.g., resume), extract here defensively.
    """
    import zipfile
    for root, _, files in os.walk(round_dir):
        for fname in files:
            if re.search(r"validation-\d+\.zip$", fname):
                zpath = os.path.join(root, fname)
                try:
                    with zipfile.ZipFile(zpath, "r") as z:
                        z.extractall(root)
                except Exception:
                    pass


def load_round(profiler, round_idx):
    """Load plan_stat_pairs for an already-collected round_{round_idx}.

    Replaces the previous Profiler.parse_metrics_from_zip path. That path had
    non-deterministic action↔metrics pairing under ThreadPoolExecutor — pairs
    sometimes ended up with metrics from a different plan than their action.
    This implementation:
      - walks chunk_*/plans/plan_*.h5 in sorted (deterministic) order
      - for each plan, reads its matching stats-plan_*.txt file directly
      - parses with process_trace_all and computes per-group median nonzero
      - returns the same tuple format: (actions, avg_list, med_list, std_list, wall_s)
    """
    from collections import defaultdict
    from mimesys.preprocessing.dataloader import parse_trace_file, process_trace_all

    dest = os.path.join(OUTPUT_PATH, f"round_{round_idx}")
    if not os.path.isdir(dest):
        return []
    # Defensively extract any validation zips not yet expanded.
    _ensure_extracted(dest)

    plan_paths = sorted(__import__("glob").glob(f"{dest}/chunk_*/plans/plan_*.h5"))
    if not plan_paths:
        return []
    print(f"  loading round_{round_idx}: {len(plan_paths)} plans from disk (clean parse)")

    pairs = []
    for h5p in plan_paths:
        sp = h5p.replace("/plans/plan_", "/results/stats-plan_").replace(".h5", ".txt")
        if not os.path.exists(sp):
            continue
        try:
            _, traces = parse_trace_file(sp)
            pm = process_trace_all(traces, include_aggregated_cpu=True)
        except Exception:
            continue
        if not pm:
            continue
        with h5py.File(h5p, "r") as f:
            actions = f["execution_plan"][:].tolist()
        num_actions = len(actions) + 1

        # Per-group median of nonzero, per metric. Group i collects samples
        # at sample_idx % num_actions == i (samples cycle through prev / curr /
        # noop slots in the BM_Mimesys loop).
        avg_metrics = defaultdict(dict)
        med_metrics = defaultdict(dict)
        std_metrics = defaultdict(dict)
        for tm, vals in pm.items():
            groups = defaultdict(list)
            for i, v in enumerate(vals):
                groups[i % num_actions].append(v)
            for g, gvals in groups.items():
                nz = [x for x in gvals if x != 0]
                avg_metrics[g][tm] = (sum(nz) / len(nz)) if nz else 0.0
                med_metrics[g][tm] = sorted(nz)[len(nz) // 2] if nz else 0.0
                std_metrics[g][tm] = float(np.std(nz)) if nz else 0.0

        # Flatten as the old code did: outer-group then inner-metric order
        # (metric order = process_trace_all dict insertion order, deterministic).
        flat_avg, flat_med, flat_std = [], [], []
        keys = list(pm.keys())
        for g in range(num_actions):
            for tm in keys:
                flat_avg.append(avg_metrics[g].get(tm, 0.0))
                flat_med.append(med_metrics[g].get(tm, 0.0))
                flat_std.append(std_metrics[g].get(tm, 0.0))

        try:
            ts0 = float(traces[0]["timestamp"][0].split()[0])
            ts1 = float(traces[-1]["timestamp"][0].split()[0])
            wall_s = max(0.0, ts1 - ts0)
        except Exception:
            wall_s = float("nan")

        pairs.append((actions, flat_avg, flat_med, flat_std, wall_s))
    return pairs


def pairs_to_AM(pairs, max_len=0):
    """Convert plan_stat_pairs → (A list, M array, var array) with padding.
    Tolerates pairs of length 4 (legacy) or 5 (new — wall-clock tucked at
    index 4)."""
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
# Straggler filter: drop plans whose wall-clock blew the slot budget
# ---------------------------------------------------------------------------

# Worker side: each plan runs for MIMESYS_ITERS × (T+1) slots × slot_seconds.
# With T=2 timesteps, do_sleep=true, MIMESYS_ITERS=3, slot=1 s ⇒ ~9 s budget.
# Add 2 s for benchmark startup + tmp-file cleanup per plan.
EXPECTED_PLAN_SECONDS = 18.0  # T=2 timesteps × MIMESYS_ITERS=3 × slot=2s = 18s
PLAN_OVERHEAD_SECONDS = 2.0
STRAGGLER_THRESHOLD   = 1.5   # drop plans > 1.5× expected


def filter_straggler_plans(pairs, expected_s=EXPECTED_PLAN_SECONDS,
                            overhead_s=PLAN_OVERHEAD_SECONDS,
                            threshold=STRAGGLER_THRESHOLD):
    """Drop plan_stat_pairs whose wall-clock (5th tuple element) exceeded
    `(expected_s + overhead_s) × threshold`. Pairs without wall-clock (legacy
    4-tuples) are passed through.
    Returns (kept_pairs, dropped_count, max_seen_seconds)."""
    if not pairs:
        return pairs, 0, 0.0
    budget_s = (expected_s + overhead_s) * threshold
    kept, dropped, max_seen = [], 0, 0.0
    for p in pairs:
        if p is None:
            continue
        if len(p) < 5 or p[4] != p[4]:    # NaN check
            kept.append(p); continue
        wall_s = float(p[4])
        max_seen = max(max_seen, wall_s)
        if wall_s > budget_s:
            dropped += 1
        else:
            kept.append(p)
    print(f"  [straggler-filter] kept {len(kept)}/{len(pairs)} "
          f"(dropped {dropped} with wall_s > {budget_s:.1f} s; "
          f"max_seen={max_seen:.1f} s, expected~{expected_s+overhead_s:.1f} s)")
    return kept, dropped, max_seen


# ---------------------------------------------------------------------------
# Metric summary
# ---------------------------------------------------------------------------

def report_metrics(M, label=""):
    labels = ["CPU%", "IO KB/s", "L3 MB/s", "BW %"]
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
        # Dispatch + extract zips on workers; skip the buggy in-line parse.
        asyncio.run(profiler.profile_actions(dest0, skip_parsing=True))
        # Clean parse from on-disk raw stats.
        pairs0 = load_round(profiler, 0)
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

    # ── Load any existing rounds 1+ (apply same filters as live runs) ─────────
    A, M, var = list(A0), M0, var0
    for r_idx in sorted(r for r in existing if r > 0):
        pairs = load_round(profiler, r_idx)
        pairs, n_dropped, _ = filter_straggler_plans(pairs)
        A_r, M_r, var_r = pairs_to_AM(pairs, max_len=M.shape[1])
        if M_r is None:
            continue
        # Pad width if needed
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

    start_round = (max(existing) + 1) if existing else 1
    print(f"\nStarting active-learning from round {start_round} "
          f"(dataset: {len(A)} samples)")

    # ── Active-learning rounds 1..n_rounds ────────────────────────────────────
    for r in range(start_round, n_rounds + 1):
        print(f"\n=== Round {r}  (batch={BATCH_SIZE}, strategy=hull:fps 5:5) ===")

        batch = propose_convex_hull_and_novelty_mix(A, M, BATCH_SIZE)

        dest = os.path.join(OUTPUT_PATH, f"round_{r}")
        write_actions_to_execution_plans(batch, dest, PROFILING_MACHINES)

        # Dispatch + extract zips on workers; skip the buggy in-line parse.
        asyncio.run(profiler.profile_actions(dest, skip_parsing=True))
        # Clean parse from on-disk raw stats.
        pairs = load_round(profiler, r)
        # Drop straggler plans (wall-clock > 1.5× expected slot budget)
        pairs, n_dropped, max_seen = filter_straggler_plans(pairs)
        A_r, M_r, var_r = pairs_to_AM(pairs, max_len=M.shape[1])
        n_valid = 0 if M_r is None else len(A_r)
        print(f"  Valid: {n_valid}/{len(pairs) + n_dropped}  (stragglers dropped: {n_dropped})")

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

    print(f"collect_training_data: round 0 (initial_candidates) + {args.rounds} "
          f"active rounds → {OUTPUT_PATH}"
          + (" [RESTART]" if args.restart else ""))
    run_collection(n_rounds=args.rounds, restart=args.restart)
