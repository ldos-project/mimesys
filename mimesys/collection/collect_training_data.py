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
# Env-var override: MIMESYS_WORKER_CONFIG_PATH lets you point at any config
# file (including one with a space in its name like "config copy.py"). Falls
# back to the regular `import config` from worker_scripts/.
_worker_cfg_path = os.environ.get("MIMESYS_WORKER_CONFIG_PATH")
if _worker_cfg_path:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("worker_config", _worker_cfg_path)
    worker_config = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(worker_config)
    print(f"[collect] worker_config loaded from {_worker_cfg_path}")
else:
    import config as worker_config

OUTPUT_PATH = os.path.expanduser(
    os.environ.get("MIMESYS_OUTPUT_PATH",
                    "~/mimesys_training_data/training_data_v2_2sec"))

PROFILING_MACHINES = list(worker_config.HOSTNAMES)
SSH_USER           = worker_config.USERNAME
SSH_KEY_PATH       = os.path.expanduser(worker_config.PRIVATE_KEY_PATH)
MY_HOSTNAME        = worker_config.MY_HOSTNAME

NUM_ACTIONS    = int(os.environ.get("MIMESYS_NUM_ACTIONS", "19"))
NUM_THREADS    = 20
BATCH_SIZE     = int(os.environ.get("MIMESYS_BATCH_SIZE",
                     worker_config.PER_MACHINE_BATCH * len(PROFILING_MACHINES)))   # active-learning round size (env-overridable)
# Per active round, for each none-current curr we also collect K prev-variants
# [prev, curr]. Prevs are drawn PER-CURR at random from an FPS shortlist of
# ~K*PREV_SHORTLIST_MULT none-current pool actions (the shortlist spans the metric
# space → high LLC / mem-BW prevs stay represented). 0 disables variants. See
# build_prev_variants.
K_PREV_VARIANTS = int(os.environ.get("MIMESYS_PREV_VARIANTS", "2"))
PREV_SHORTLIST_MULT = int(os.environ.get("MIMESYS_PREV_SHORTLIST_MULT", "10"))
# Strategy (a): concentrate prevs in the high-LLC regime (carry-over is strongest
# there; low-LLC prevs only add noise) and emit each (prev,curr) as N independent
# replicate plans so the carry-over Δ beats the per-plan measurement noise.
PREV_N_REP = int(os.environ.get("MIMESYS_PREV_N_REP", "3"))
# Replication count for none-current plans (pure AL, no prev-variant). Each
# unique curr is profiled N_REP times so noise averaging can beat per-plan
# measurement variance. Default 1 (legacy: each curr profiled once).
N_REP = int(os.environ.get("MIMESYS_N_REP", "1"))
PREV_HIGH_LLC_FRAC = float(os.environ.get("MIMESYS_PREV_HIGH_LLC_FRAC", "0.5"))  # top fraction of pool by LLC
# Selection mode for prev_action candidate filter:
#   "llc"           — top PREV_HIGH_LLC_FRAC of pool by LLC only (legacy default)
#   "high_resource" — union of top frac by LLC, BW, and IO + random sample
#                     (mixes all carry-over types; IO carry-over is the largest
#                      effect for the new HddRead/HddWriteNF stressors)
PREV_FILTER_MODE = os.environ.get("MIMESYS_PREV_FILTER_MODE", "llc")
PREV_RANDOM_FRAC = float(os.environ.get("MIMESYS_PREV_RANDOM_FRAC", "0.15"))   # fraction of pool to add as random
# 31 = 28 hpcperfstatsd + 3 pqos (single merged group)
# Layout per group (offsets within a 31-block), alphabetical from process_trace_all
# then pqos appended in PQOS_METRIC_KEYS order:
#   0-19   per-core CPU% (avg_cpu_utilizations_core_00..19)
#   20     avg_cpu_utilizations_total
#   21     io                       (read + write, summary for bucketing)
#   22     io_read
#   23     io_write
#   24     l3_cache_usage           (CHA aggregate)
#   25     memory_bandwidth         (read + write, summary for bucketing)
#   26     memory_bandwidth_read
#   27     memory_bandwidth_write
#   28     pqos_llc_kb              (LLC occupancy, unique vs hpc)
#   29     pqos_ipc                 (unique vs hpc)
#   30     pqos_misses              (pairs with ipc for MPKI)
NUM_METRICS    = 31
# How many hpc-side metrics (pre-pqos). Used to size hpc loop in load_round.
NUM_HPC_METRICS = 28

# 5:5 ratio — equal stratified split between hull-cell-fill and fps-novelty.
# The hull_fps_ratio passed into propose_by_hull_mixed_fps_hybrid both (a) sizes
# the two source pools (target |hull_pool|/|fps_pool|) AND (b) sets the
# stratified-sampling target for the final n_candidates draw (50% from each
# pool when ratio=1.0). Previously 3/7 caused the merged-pool random sample
# to skew ~30% hull / 70% fps regardless of intent.
HULL_FPS_RATIO = 1.0

# ── prev=None convention ────────────────────────────────────────────────────
# This script collects NONE-CURRENT data: every plan is [NONE_PREV, curr]. The
# "None" prev is a near-idle placeholder (1 thread, tiny weight) — it keeps each
# plan 2-window so the metric slot layout (prev/curr/noop, curr=slot 1) is
# unchanged, while contributing ~zero resource footprint (≈ "no prior action").
# Paired collection (collect_paired_prev_data.py) reuses these helpers and
# swaps NONE_PREV for a real prev sampled from the none-current pool.
# "No previous action" = an all-zero prev window. write_actions drops all-zero
# windows, so a [NONE_PREV, curr] action is written to the H5 as curr-only (one
# window, ≈ a third less profiling time). load_round / the dataloader reconstruct
# the 2-window [NONE_PREV, curr] action and a zero prev metric slot at read time.
NONE_PREV = [[0.0] * NUM_ACTIONS for _ in range(NUM_THREADS)]

def _action_depth(action):
    """Nesting depth: 3 ⇒ 2-window [win][thread][stressor]; 2 ⇒ 1-window [thread][stressor]."""
    d = 0; x = action
    while isinstance(x, (list, tuple)) and len(x) > 0:
        d += 1; x = x[0]
    return d

def curr_window(action):
    """Return the current-action window (NUM_THREADS, NUM_ACTIONS) from an action
    that is either 1-window (depth 2) or 2-window [prev, curr] (depth 3)."""
    return action[-1] if _action_depth(action) == 3 else action

def as_none_current(action):
    """Force prev=None: return [NONE_PREV, curr] (always 2-window)."""
    return [NONE_PREV, curr_window(action)]

def ensure_two_window(action):
    """Append NONE_PREV in front if the action is single-window, so 1- and 2-stage
    collections always share the same (2-window) action dimension."""
    return action if _action_depth(action) == 3 else [NONE_PREV, action]


def sample_scaling_weight():
    weights = np.arange(90, 105, 5)
    probabilities = weights / weights.sum()
    return np.random.choice(weights, p=probabilities)

def propose_candidates_by_random(timestamp=2, num_actions=None, num_threads=20, n_candidates=1000):
    if num_actions is None:
        num_actions = NUM_ACTIONS   # defined at top of module — currently 15
    """Random candidates.

    Two modes selectable via env var ``MIMESYS_DENSE_PROPOSALS`` (Tier 3.6):
      "0" (default)  — original behavior: zeros 1..N-1 random thread rows
                       and N/2..N-1 random stressor columns per thread.
      "1"            — dense mode: zeros at most N/4 thread rows (so most
                       threads stay active) and N/4..N/2 stressor columns
                       per thread (denser per-thread stressor mix). Designed
                       to match the all-cores-busy test distribution.
    """
    candidates = []
    action_shape = (timestamp, num_threads, num_actions)
    _dense_mode = os.environ.get("MIMESYS_DENSE_PROPOSALS", "0") == "1"

    def zero_out_random_rows(mutated):
        if _dense_mode:
            max_rows_to_zero = max(0, len(mutated) // 4)        # 0..N/4 ≈ 0..5 of 20
        else:
            max_rows_to_zero = max(1, len(mutated) - 1)         # 1..N-1 ≈ 1..19 of 20
        num_rows_to_zero = random.randint(0 if _dense_mode else 1, max_rows_to_zero)
        if num_rows_to_zero == 0:
            return mutated
        rows_to_zero = random.sample(range(len(mutated)), num_rows_to_zero)
        for row_idx in rows_to_zero:
            mutated[row_idx] = [0.0] * len(mutated[row_idx])
        return mutated

    for _ in range(n_candidates):
        candidate = []
        for _ in range(action_shape[0]):
            thread = []
            for _ in range(action_shape[1]):
                if _dense_mode:
                    # Zero out 3..6 of 13 stressors per thread → 7..10 active stressors.
                    num_cols_to_zero = random.randint(action_shape[2] // 4,
                                                       action_shape[2] // 2)
                else:
                    num_cols_to_zero = random.randint(action_shape[2] // 2,
                                                       action_shape[2] - 1)
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


def propose_candidates_by_dense_seeds(timestamp=2, num_actions=None,
                                       num_threads=20, n_candidates=1000):
    """Dense-seed proposals mirroring the OLD surrogate_based_search.py
    ``initial_candidates``: every thread is active (no idle threads, no scaling
    < 1.0). Each candidate is built by:

      1. Pick 1, 2, or 3 stressor indices uniformly at random.
      2. Allocate Dirichlet(1) weights over those stressors that sum to 1.0
         (so each thread's stressor weights sum exactly to 1.0).
      3. Replicate this single thread-vector across ALL `num_threads`, then
         shuffle the thread order (purely cosmetic — the model is exposed to
         both ordered and shuffled forms via permutation augmentation).

    Produces seeds with 20 active threads at full intensity per thread —
    matches the per-core CPU distribution of the test traces (mean ~42% on
    c220g5 instead of 22% for the standard random/hull proposals).
    """
    if num_actions is None:
        num_actions = NUM_ACTIONS
    candidates = []
    for _ in range(n_candidates):
        candidate = []
        for _ in range(timestamp):
            # Choose 1-3 dominant stressors; full-weight Dirichlet.
            n_dom   = random.choice([1, 1, 2, 3])
            dom_idx = random.sample(range(num_actions), n_dom)
            dirichlet_w = np.random.dirichlet(np.ones(n_dom))
            base_thread = [0.0] * num_actions
            for w, idx in zip(dirichlet_w, dom_idx):
                base_thread[idx] = float(w)
            # Replicate base thread across all 20 cores; small per-thread
            # noise on the weight so we don't produce literally identical rows.
            threads = []
            for _ in range(num_threads):
                noisy = [v * float(np.random.uniform(0.9, 1.0)) for v in base_thread]
                # Renormalize so thread sum stays in [0, 1].
                s = sum(noisy)
                if s > 1.0:
                    noisy = [v / s for v in noisy]
                threads.append(noisy)
            random.shuffle(threads)
            candidate.append(threads)
        candidates.append(candidate)
    return candidates


def propose_candidates_by_thread_sparsity(timestamp=2, num_actions=None,
                                           num_threads=20, n_candidates=1000):
    """Structured thread-sparse candidates: K threads at high utilization, rest idle.

    Fills the gap where the existing random/composite/hull-FPS proposals tend to
    spread activity across all threads.  Each candidate picks:
      - K  ∈ {1, 2, 3, 4, 5, 6, 7, 8, 10, 12}  (biased toward small K)
      - layout ∈ {first_K, last_K, random_K, socket0_K, socket1_K}
      - per active thread: 1–2 dominant stressors at high scale (~0.9–1.0)
      - all inactive threads forced to 0.

    Returns the same nested-list shape as ``propose_candidates_by_random``:
    list[n_candidates] of list[timestamp][num_threads][num_actions].
    """
    if num_actions is None:
        num_actions = NUM_ACTIONS
    K_choices = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    K_probs   = [0.10, 0.15, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05]
    layouts   = ["first_K", "last_K", "random_K", "socket0_K", "socket1_K"]
    half      = num_threads // 2

    def gen_active_thread():
        # 1-2 dominant stressors, weights summing to 1, scaled by 0.9-1.0.
        n_dom   = random.choice([1, 1, 2])    # bias single-stressor cores
        dom_idx = random.sample(range(num_actions), n_dom)
        action  = [0.0] * num_actions
        dirichlet_w = np.random.dirichlet(np.ones(n_dom))
        scale = sample_scaling_weight() / 100.0    # 0.9 / 0.95 / 1.0
        for w, idx in zip(dirichlet_w, dom_idx):
            action[idx] = float(w) * scale
        return action

    candidates = []
    for _ in range(n_candidates):
        candidate = []
        for _ in range(timestamp):
            K      = int(np.random.choice(K_choices, p=K_probs))
            layout = random.choice(layouts)
            if layout == "first_K":
                active = list(range(K))
            elif layout == "last_K":
                active = list(range(num_threads - K, num_threads))
            elif layout == "random_K":
                active = random.sample(range(num_threads), K)
            elif layout == "socket0_K":
                K_eff  = min(K, half)
                active = random.sample(range(0, half), K_eff)
            else:  # socket1_K
                K_eff  = min(K, half)
                active = random.sample(range(half, num_threads), K_eff)
            active_set = set(active)

            threads = []
            for t in range(num_threads):
                if t in active_set:
                    threads.append(gen_active_thread())
                else:
                    threads.append([0.0] * num_actions)
            candidate.append(threads)
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

    # ── operator 7: socket remove — zero out ALL 10 threads on one socket ───
    # Produces extreme S0:S1 asymmetry (one socket idle, the other working).
    # Useful for sampling the per-socket CPU corners (cpu-s0=0 × cpu-s1=high,
    # and vice versa) that other operators tend to miss.
    def _socket_remove(a_arrs):
        def op(curr):
            n_th  = curr.shape[0]
            split = n_th // 2
            result = curr.copy()
            # Pick which socket to zero (0 = S0, 1 = S1)
            zero_socket = random.randint(0, 1)
            if zero_socket == 0:
                result[0:split] = 0.0
            else:
                result[split:n_th] = 0.0
            return result
        return _apply_to_curr(a_arrs, op)

    operators = ["thread_swap", "stressor_crossover", "blend", "scale_noise",
                 "socket_crossover", "socket_scale", "socket_remove"]
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
            elif op == "socket_remove":
                result_arrs = _socket_remove(a1)
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
# Within each NUM_METRICS-dim group, hpc metrics are at offsets 0..NUM_HPC_METRICS-1
# (the "first 28 cols"), and the 4 *key* hpc metrics for bucketing/FPS are:
#   idx+20  avg_cpu_utilizations_total
#   idx+21  io                  (summary sum: io_read + io_write)
#   idx+24  l3_cache_usage      (CHA aggregate)
#   idx+25  memory_bandwidth    (summary sum: bw_read + bw_write)
# pqos metrics live at offsets NUM_HPC_METRICS..NUM_METRICS-1 (offsets 28..30).
# For curr_action (group 1, base = NUM_METRICS): key metrics at base+{20,21,24,25}.
# The 4 indices map to [CPU%, IO KB/s, L3 MB/s, BW %] as labeled in report_metrics.
_HPC_KEY_OFFSETS = [20, 21, 24, 25]   # within a NUM_METRICS-block, the 4 summary indices
_KEY_METRIC_INDICES = [NUM_METRICS + off for off in _HPC_KEY_OFFSETS]


# ---------------------------------------------------------------------------
# prev-action variant sampling (paired curr × prev data, collected inline)
# ---------------------------------------------------------------------------

def _fps_indices(X, k):
    """Farthest-point sampling: k row indices of X spread across the normalized
    metric space (maximin → reaches the extremes, e.g. high LLC / mem-BW)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n <= k:
        return list(range(n))
    mn, mx = X.min(0), X.max(0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    Xn = (X - mn) / rng
    chosen = [int(np.argmax(np.linalg.norm(Xn - Xn.mean(0), axis=1)))]  # farthest from centroid
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(Xn - Xn[c], axis=1) for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    return chosen


def _prev_is_zero(action):
    """True if a (2-window) action's prev window is all-zero — i.e. a none-current
    sample. After load_round, none-current plans carry prev=NONE_PREV(=0); real
    prev-variants carry a nonzero prev window."""
    return len(action) >= 2 and all(all(v == 0.0 for v in row) for row in action[0])


def _curr_is_empty(action, eps=1e-6):
    """True if the curr (last) window has effectively no work (total weight ≤ eps).
    Such a candidate would profile an idle plan (and, with NONE_PREV=0, used to write
    an empty H5). Filtered at proposal time so we never waste a plan on a no-op."""
    return sum(v for row in curr_window(action) for v in row) <= eps


def build_prev_variants(currs, A_pool, M_pool, k=K_PREV_VARIANTS,
                        shortlist_mult=PREV_SHORTLIST_MULT, n_rep=PREV_N_REP,
                        high_llc_frac=PREV_HIGH_LLC_FRAC, seed=0):
    """For each curr action, build variant plans [prev, curr] (strategy a):
      1. restrict prev candidates to the top `high_llc_frac` of the pool by LLC —
         carry-over is strongest there; low-LLC prevs only add noise;
      2. FPS a diverse shortlist (~k*shortlist_mult) within that high-LLC subset;
      3. draw k prevs at random per curr (per-sample variety);
      4. emit each (prev, curr) as `n_rep` INDEPENDENT replicate plans so the
         carry-over Δ can be medianed across plans, beating per-plan noise.
    The pool is none-current only → no prev-prev contamination. Returns
    (variants, shortlist_footprints); ([], []) if pool empty or k<=0."""
    if k <= 0 or A_pool is None or len(A_pool) == 0 or M_pool is None or len(M_pool) == 0:
        return [], []
    M_pool_np = np.asarray(M_pool)
    if PREV_FILTER_MODE == "high_resource":
        # Take top `high_llc_frac` of pool by EACH of LLC, BW, IO; union them;
        # then add a random sample of size `PREV_RANDOM_FRAC × len(pool)`.
        # _KEY_METRIC_INDICES = [CPU_total, IO, LLC, BW] for curr-group metrics.
        io_col, llc_col, bw_col = _KEY_METRIC_INDICES[1], _KEY_METRIC_INDICES[2], _KEY_METRIC_INDICES[3]
        keep = max(k, int(np.ceil(len(M_pool_np) * high_llc_frac)))
        top_llc = set(np.argsort(M_pool_np[:, llc_col])[-keep:].tolist())
        top_bw  = set(np.argsort(M_pool_np[:, bw_col])[-keep:].tolist())
        top_io  = set(np.argsort(M_pool_np[:, io_col])[-keep:].tolist())
        n_rand  = max(0, int(np.ceil(len(M_pool_np) * PREV_RANDOM_FRAC)))
        rand_idx = random.Random(seed + 7).sample(range(len(M_pool_np)),
                                                  min(n_rand, len(M_pool_np)))
        hi_idx = np.array(sorted(top_llc | top_bw | top_io | set(rand_idx)), dtype=int)
    else:
        # Legacy "llc" mode: top fraction by LLC only.
        llc = M_pool_np[:, _KEY_METRIC_INDICES[2]]
        keep = max(k, int(np.ceil(len(llc) * high_llc_frac)))
        hi_idx = np.argsort(llc)[-keep:]
    # 2. diverse FPS shortlist within the high-LLC subset.
    n_short = min(len(hi_idx), max(k, k * shortlist_mult))
    short_local = _fps_indices(M_pool[hi_idx][:, _KEY_METRIC_INDICES], n_short)
    short_idxs = [int(hi_idx[i]) for i in short_local]
    rng = random.Random(seed)
    variants = []
    for c in currs:
        cw = curr_window(c)
        for i in rng.sample(short_idxs, min(k, len(short_idxs))):   # 3. per-curr random subset
            for _ in range(n_rep):                                   # 4. N replicate plans
                variants.append([curr_window(A_pool[i]), cw])
    foot = [[round(float(M_pool[i, j]), 1) for j in _KEY_METRIC_INDICES] for i in short_idxs]
    return variants, foot


def _split_none_current(pairs):
    """Partition load_round pairs into (none_current, prev_variant) by prev window."""
    valid = [p for p in pairs if p is not None]
    none_cur = [p for p in valid if _prev_is_zero(p[0])]
    variant  = [p for p in valid if not _prev_is_zero(p[0])]
    return none_cur, variant


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
            base = K * NUM_METRICS                      # 0, 31, 62  (NUM_METRICS=31)
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

    Hull dims (built from curr_action group of NUM_METRICS=31 layout):
                    0=CPU%      (avg_cpu_utilizations_total)
                    1=IO        (io summary = io_read + io_write)
                    2=L3        (l3_cache_usage — traffic to L3, MB/s)
                    3=BW        (memory_bandwidth summary = bw_read + bw_write)
                    4=CPU-S0    (per-socket CPU util, mean of cols base+0..9)
                    5=CPU-S1    (per-socket CPU util, mean of cols base+10..19)
                    6=PQOS-LLC  (LLC occupancy in KB — orthogonal to L3 traffic;
                                  added with new merged-group pqos panel)
                    7=PQOS-IPC  (instructions per cycle — compute-efficiency
                                  axis distinct from CPU% which is util only)
                    8=PQOS-MPKI (LLC misses per kilo-instruction — derived from
                                  pqos_misses and pqos_ipc; cache-miss rate
                                  normalized for workload intensity)
                    9=IO_READ   (disk read traffic, KB/s — split from summary IO)
                   10=IO_WRITE  (disk write traffic, KB/s — split from summary IO)
                   11=BW_READ   (memory read bandwidth %, split from summary BW)
                   12=BW_WRITE  (memory write bandwidth %, split from summary BW)
    Index legend supersedes the prior "23-D" stub since the metric panel grew
    to 31 with hpc r/w splits and 3 pqos fields.
    """
    from collections import Counter

    # Augment pool with thread-perm + socket-swap variants so hull/FPS don't
    # treat augmentation-equivalents of existing samples as novel candidates.
    n_orig = len(A)
    A, M = _augment_pool_for_hull(A, M)
    print(f"  [hull/fps] augmented pool: {n_orig} → {len(A)} samples "
          f"({AUG_FACTOR}× via intra-perm + socket-swap variants)")

    M_key4 = M[:, _KEY_METRIC_INDICES].astype(float)
    # Per-core CPU% live in group 1 (curr_action), cols base+0..base+19
    # (alphabetical core_00..core_19). Socket 0 = cores 0..9, socket 1 = 10..19.
    G1_BASE = NUM_METRICS                 # 31 — start of curr_action block
    per_core_curr = M[:, G1_BASE:G1_BASE + 20].astype(float)
    cpu_s0 = per_core_curr[:, :10].mean(axis=1, keepdims=True)
    cpu_s1 = per_core_curr[:, 10:].mean(axis=1, keepdims=True)
    # pqos columns within the curr block (NUM_HPC_METRICS=28 hpc, then pqos):
    #   offset 28 = pqos_llc_kb, 29 = pqos_ipc, 30 = pqos_misses
    pqos_llc    = M[:, G1_BASE + 28:G1_BASE + 29].astype(float)
    pqos_ipc    = M[:, G1_BASE + 29:G1_BASE + 30].astype(float)
    pqos_misses = M[:, G1_BASE + 30:G1_BASE + 31].astype(float)
    # MPKI = LLC misses × 1000 / instructions_retired.
    # instructions ≈ ipc × cycles ≈ ipc × cpu_freq × elapsed × num_cores.
    # For our 1-sec sample window on c220g5 (Xeon Silver 4114, base 2.2 GHz, 20 cores):
    #   instructions ≈ ipc × 2.2e9 × 1.0 × 20 = ipc × 44e9
    #   MPKI ≈ misses × 1000 / (ipc × 44e9) = misses / (ipc × 44e6)
    # Clamp ipc denominator so divisions by ~0 (idle slots) don't blow up.
    ipc_safe = np.maximum(pqos_ipc, 0.05)
    pqos_mpki = (pqos_misses / (ipc_safe * 4.4e7)).astype(float)
    # r/w split dims for IO and memory_bandwidth (curr-block offsets 22,23,26,27).
    # Hull groups using these explore the read-heavy ↔ write-heavy axis which the
    # summary IO/BW dims (1, 3) flatten by summing.
    io_read  = M[:, G1_BASE + 22:G1_BASE + 23].astype(float)
    io_write = M[:, G1_BASE + 23:G1_BASE + 24].astype(float)
    bw_read  = M[:, G1_BASE + 26:G1_BASE + 27].astype(float)
    bw_write = M[:, G1_BASE + 27:G1_BASE + 28].astype(float)
    M_key = np.hstack([M_key4, cpu_s0, cpu_s1, pqos_llc, pqos_ipc, pqos_mpki,
                       io_read, io_write, bw_read, bw_write])  # (N, 13)

    # ── 1. Build hull pool ────────────────────────────────────────────────────
    # 7 groups span the metric space:
    #   (a) IO×L3×BW             — memory-system saturation (summary)
    #   (b) CPU×L3×PQOS-LLC      — cache traffic vs occupancy diversity
    #   (c) CPU×IPC×MPKI         — compute efficiency vs miss-rate joint
    #   (d) BW×MPKI              — BW vs cache thrashing
    #   (e) CPU-S0×CPU-S1        — socket asymmetry
    #   (f) IO_READ×IO_WRITE     — disk r/w mix exploration
    #   (g) BW_READ×BW_WRITE     — memory r/w mix exploration
    # Kept ≤3-D per group so Delaunay is feasible.
    GROUPS = [
        (1, 2, 3),       # IO × L3 × BW             (memory-system saturation)
        (0, 2, 6),       # CPU × L3 × PQOS-LLC      (cache traffic vs occupancy)
        (0, 7, 8),       # CPU × IPC × MPKI         (compute efficiency vs miss rate)
        (3, 8),          # BW × MPKI                (BW vs cache thrashing)
        (4, 5),          # CPU-S0 × CPU-S1          (socket asymmetry)
        (9, 10),         # IO_READ × IO_WRITE       (NEW: disk r/w mix)
        (11, 12),        # BW_READ × BW_WRITE       (NEW: memory r/w mix)
    ]

    n_str   = len(A[0][0][0])
    hull_pool = []

    for dims in GROUPS:
        d = len(dims)
        M_sub = M_key[:, list(dims)]
        mn, mx = M_sub.min(0), M_sub.max(0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        M_norm = (M_sub - mn) / rng

        # When MIMESYS_BBOX_HULL=1, ignore the data-convex-hull and use the full
        # [0, max] bounding box (in normalized space, [0, 1]) instead. This lets
        # us sample EMPTY corners — combinations of high metrics never observed
        # together — by elementwise-mixing the per-axis champion actions.
        use_bbox = os.environ.get("MIMESYS_BBOX_HULL", "0") == "1"

        # MIMESYS_FAST_HULL=1 skips Delaunay (exponential in d) and approximates
        # inside/outside using nearest-data-point distance. Threshold = 1.5 cell
        # widths in normalized space → a cell is "outside" if no data point sits
        # within ~1.5 cells of it.
        fast_hull = os.environ.get("MIMESYS_FAST_HULL", "0") == "1"
        tri = None
        tri_ok = False
        if not fast_hull:
            try:
                tri = Delaunay(M_norm)
                tri_ok = True
            except Exception:
                if not use_bbox:
                    print(f"  [hull_fps_hybrid] dims={dims}: Delaunay failed, skipping")
                    continue

        g = np.linspace(0.5 / grid_bins, 1.0 - 0.5 / grid_bins, grid_bins)
        grids = np.meshgrid(*[g] * d, indexing='ij')
        centres = np.stack([gg.ravel() for gg in grids], axis=1)

        # Pre-compute per-centre nearest-data distance (used by fast_hull, and
        # for cheap reuse by both inside/outside classification and priority).
        if fast_hull:
            nn_thresh = 1.5 / grid_bins
            # Chunked nearest-data distance — avoid OOM for grid_bins^d × N matrix.
            n_centres = len(centres)
            nn_dist = np.empty(n_centres, dtype=np.float32)
            chunk = max(1, min(8192, n_centres))
            for i0 in range(0, n_centres, chunk):
                blk = centres[i0:i0 + chunk]                              # (B, d)
                cd  = np.linalg.norm(blk[:, None, :] - M_norm[None, :, :], axis=2)
                nn_dist[i0:i0 + chunk] = cd.min(axis=1)
            centre_outside = nn_dist > nn_thresh
        else:
            nn_dist = None
            centre_outside = None

        if use_bbox:
            inside_centres = centres   # use ALL bbox cells (no hull filter)
            inside_outside = centre_outside if fast_hull else None
        elif fast_hull:
            inside_mask = ~centre_outside
            inside_centres = centres[inside_mask]
            inside_outside = np.zeros(len(inside_centres), dtype=bool)
            if len(inside_centres) == 0:
                continue
        else:
            inside_mask = tri.find_simplex(centres) >= 0
            inside_centres = centres[inside_mask]
            inside_outside = None
            if len(inside_centres) == 0:
                continue

        # When MIMESYS_BBOX_HULL_EMPTY_ONLY=1, restrict candidates to cells that
        # are empty (no existing pool sample). This concentrates the candidate
        # budget on true exploration regions rather than re-blending around
        # already-covered cells.
        empty_only = os.environ.get("MIMESYS_BBOX_HULL_EMPTY_ONLY", "0") == "1"
        if empty_only and len(inside_centres) > 0:
            def _cell_of_pt(c):
                return tuple(np.clip((c * grid_bins).astype(int), 0, grid_bins - 1).tolist())
            from collections import Counter as _Counter
            _pre_counts = _Counter(map(tuple,
                np.clip((M_norm * grid_bins).astype(int), 0, grid_bins - 1).tolist()))
            keep_idx = [i for i, c in enumerate(inside_centres)
                        if _pre_counts.get(_cell_of_pt(c), 0) == 0]
            inside_centres = inside_centres[keep_idx] if len(keep_idx) else np.empty((0, d))
            if inside_outside is not None and len(inside_outside) > 0:
                inside_outside = inside_outside[keep_idx]
            if len(inside_centres) == 0:
                print(f"  [hull_fps_hybrid] dims={dims}: no empty cells (all covered), skipping")
                continue

        def cell_of(pts):
            return np.clip((pts * grid_bins).astype(int), 0, grid_bins - 1)

        cell_counts = Counter(map(tuple, cell_of(M_norm).tolist()))

        def priority(c):
            ci   = tuple(cell_of(c[None])[0].tolist())
            cnt  = cell_counts.get(ci, 0)
            dist = float(np.linalg.norm(M_norm - c, axis=1).min())
            return (cnt, -dist)

        order = sorted(range(len(inside_centres)),
                       key=lambda k: priority(inside_centres[k]))
        sorted_centres = inside_centres[order]
        sorted_outside = (inside_outside[order]
                          if inside_outside is not None and len(inside_outside) > 0
                          else None)

        # For BBOX mode, pre-compute per-axis champions of this subset of dims.
        # When the target cell sits OUTSIDE the data hull, build the candidate by
        # ELEMENTWISE MAX of the axis champions (each axis weighted by its target
        # value in the cell) — this combines the stressors that drive each
        # metric's max, attempting to reach corners.
        if use_bbox:
            axis_champ_actions = [A[int(np.argmax(M_norm[:, di]))] for di in range(d)]
        before = len(hull_pool)
        for ci, pt in enumerate(sorted_centres):
            outside = False
            if use_bbox:
                if fast_hull and sorted_outside is not None:
                    outside = bool(sorted_outside[ci])
                elif tri_ok:
                    outside = (tri.find_simplex(pt[None])[0] < 0)
                else:
                    outside = True

            if use_bbox and outside:
                # Elementwise-max blend of axis champions, weighted by the target
                # value of the cell in each axis (target ∈ [0,1] in M_norm space).
                a_arrs = [np.asarray(a[-1]) for a in axis_champ_actions]  # curr only
                max_th = max(arr.shape[0] for arr in a_arrs)
                target_w = np.clip(pt, 0.0, 1.0)
                target_w = target_w / max(target_w.sum(), 1e-6)   # normalize to weights
                ts_out = []
                for th in range(max_th):
                    sv = np.zeros(n_str)
                    for di, arr in enumerate(a_arrs):
                        if th < arr.shape[0]:
                            row = arr[th] * (0.5 + target_w[di])  # weighted scale
                            sv = np.maximum(sv, row)
                    s = sv.sum()
                    if s > 1.0: sv /= s
                    ts_out.append(sv.tolist())
                result = [ts_out]   # curr-only (1 window) plan
            else:
                # Standard k-NN weighted blend
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
        n_out = (int(sorted_outside.sum()) if sorted_outside is not None else 0)
        tag = "fast" if fast_hull else ("delaunay" if tri_ok else "no-tri")
        print(f"  [hull_fps_hybrid] dims={dims} [{tag}]: {len(inside_centres)} cells "
              f"({n_empty} empty, {n_out} outside) → +{len(hull_pool) - before} candidates")

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

    # ── 3. Stratified sample: hull_fps_ratio sets the split ─────────────────
    # ratio = |hull_sample| / |fps_sample|.  e.g. 1.0 → 50/50 (5:5).
    # If either pool is short, the deficit spills to the other pool so we
    # always return n_candidates total when at least one is non-empty.
    total_pool = len(hull_pool) + len(fps_pool)
    if total_pool == 0:
        print("  [hull_fps_hybrid] empty pool, falling back to random")
        return propose_candidates_by_random(n_candidates=n_candidates)

    target_hull = int(round(n_candidates * hull_fps_ratio / (hull_fps_ratio + 1.0)))
    target_fps  = n_candidates - target_hull
    take_hull = min(target_hull, len(hull_pool))
    take_fps  = min(target_fps,  len(fps_pool))
    # Spill any deficit to the other pool.
    deficit = n_candidates - take_hull - take_fps
    if deficit > 0:
        take_hull += min(deficit, len(hull_pool) - take_hull)
        deficit = n_candidates - take_hull - take_fps
    if deficit > 0:
        take_fps += min(deficit, len(fps_pool) - take_fps)
    print(f"  [hull_fps_hybrid] stratified sample (ratio={hull_fps_ratio:.2f}): "
          f"hull {take_hull}/{len(hull_pool)} + fps {take_fps}/{len(fps_pool)} "
          f"= {take_hull + take_fps} of {n_candidates} requested")

    hull_idx = np.random.choice(len(hull_pool), size=take_hull, replace=False) \
                 if take_hull and hull_pool else []
    fps_idx  = np.random.choice(len(fps_pool),  size=take_fps,  replace=False) \
                 if take_fps  and fps_pool  else []
    out = [hull_pool[i] for i in hull_idx] + [fps_pool[i] for i in fps_idx]
    return out


def initial_candidates(bounds, n_candidates, num_max_threads=20):
    """ Generate initial candidates as one-hot vectors within the given bounds.

    Two modes selectable via env var ``MIMESYS_INITIAL_OLD_STYLE`` (default 0):
      "0" — current "sweep" mode: num_threads ∈ [1, candidate_max_threads],
            weight ∈ [0.2, 0.4, 0.6, 0.8, 1.0]. Produces ~2000 sparse-dominant
            seeds per round 0.
      "1" — old surrogate_based_search.py style: only num_threads=candidate_max
            (or 1 for IO-bound heuristic) and weight=1.0. Produces ~70 dense
            seeds per round 0 — should match the test per-core CPU profile.
    """
    old_style = os.environ.get("MIMESYS_INITIAL_OLD_STYLE", "0") == "1"
    k = len(bounds)
    candidates = []
    zero_action_weights = [0.0] * k
    for i in range(n_candidates):
        initial_action_weights = [0.0] * k
        for j in range(k):
            if i == j:
                initial_action_weights[j] = 1.0

# Unified sweep — every stressor (including IO indices 10..12) gets the
        # same num_threads × weight grid, with a single shuffled thread layout
        # per combination. The earlier i>=10 heuristic capped IO stressors at
        # max_threads=4 (or 1 in old_style) and compensated with extra shuffles
        # at lower thread counts; removing that bias gives a uniform [1..20] ×
        # weights sweep across all 13 stressors so the per-core CPU% distribution
        # is shaped by the workload itself, not by a hand-tuned per-stressor cap.
        candidate_max_threads = num_max_threads

        if old_style:
            # Single iter at max threads, single weight (1.0)
            num_threads_range = [candidate_max_threads]
            weight_range = [1.0]
        else:
            num_threads_range = range(1, candidate_max_threads + 1)
            # Default: 4 weights (0.25, 0.5, 0.75, 1.0). Override via
            # MIMESYS_INITIAL_WEIGHT_RANGE (comma-separated). e.g. "1.0"
            # restricts the sweep to weight=1.0 only across the same
            # num_threads sweep (1..candidate_max_threads).
            _wr_env = os.environ.get("MIMESYS_INITIAL_WEIGHT_RANGE", "")
            if _wr_env:
                weight_range = [float(x) for x in _wr_env.split(",") if x.strip()]
            else:
                weight_range = [0.25, 0.5, 0.75, 1.0]

        for num_threads in num_threads_range:
            candidate = [initial_action_weights for _ in range(num_threads)]
            while len(candidate) < num_max_threads:
                candidate.append(zero_action_weights)

            for weight in weight_range:
                scaled_candidate = [[w * weight for w in thread] for thread in candidate]
                random.shuffle(scaled_candidate)
                candidates.append(scaled_candidate)

    final_candidates = []
    weight = 1.0
    for action in candidates:
        scaled_action = [[w * weight for w in thread] for thread in action]
        # none-current: prev = NONE_PREV (was [scaled_action, scaled_action] = prev=curr)
        final_candidates.append([NONE_PREV, scaled_action])

    return final_candidates


def write_to_hdf5(action_weights, file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('execution_plan', data=action_weights)


def write_actions_to_execution_plans(actions, destination_path: str, profiling_machines: list[str]):
    """Write each action to a per-chunk plan H5. All-zero windows are dropped, so a
    none-current action [NONE_PREV(=zeros), curr] is written as a curr-only plan
    (one window) — the worker skips profiling the empty prev. Readers reconstruct
    [NONE_PREV, curr] + a zero prev metric slot from such curr-only plans."""
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
            # Drop a leading all-zero (none) prev window so none-current plans become
            # curr-only and skip profiling the empty prev. But NEVER drop the curr
            # (last) window or emit an empty plan: an empty H5 is not 3D and
            # segfaults the benchmark ("Dataset is not 3D").
            windows = list(action)
            while len(windows) > 1 and all(all(v == 0.0 for v in t) for t in windows[0]):
                windows.pop(0)
            write_to_hdf5(windows, file_path)


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

        # none-current: prev = NONE_PREV, curr = composite (was prev = composite)
        action = [NONE_PREV, composite.tolist()]
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


def _apply_thread_count(candidate, K):
    """Force `candidate` (timestamp × threads × stressors) to have EXACTLY K
    active threads in every window, with the active-thread budget split
    between the two sockets via a uniform-random socket assignment:

      N_s0 ~ Uniform({max(0, K-10), ..., min(10, K)})    inclusive
      N_s1 = K - N_s0

    `N_s0` random positions are then chosen on socket 0 (cores 0..9) and
    `N_s1` random positions on socket 1 (cores 10..19). This covers every
    valid (N_s0, N_s1) socket split with equal probability — for K=10 that
    includes the balanced 5/5, the 4/6 / 6/4, ..., and the extreme 0/10 /
    10/0 — instead of only the two endpoints a binary socket-flip would
    produce. Aggregated socket load is balanced by symmetry.

    The active rows themselves are sampled from the candidate's own nonzero
    rows (the stressor mix chosen by the upstream proposer) and cycled to
    fill K positions exactly — works whether upstream had fewer or more
    active rows than K. If a window has no active rows, K positions are
    synthesized as pure single-stressor threads at the standard scaling
    weight.

    `N_s0` and the position choice are sampled ONCE per candidate (same for
    both prev and curr windows) so the active-position set is coherent
    across the timestep.

    Returns a NEW candidate (does not mutate the input)."""
    n_thrd_total = NUM_THREADS
    socket_size  = n_thrd_total // 2
    n_s0_min = max(0, K - socket_size)
    n_s0_max = min(socket_size, K)
    N_s0 = random.randint(n_s0_min, n_s0_max)
    N_s1 = K - N_s0
    pos_s0 = random.sample(range(0, socket_size), N_s0) if N_s0 > 0 else []
    pos_s1 = random.sample(range(socket_size, n_thrd_total), N_s1) if N_s1 > 0 else []
    active_positions = sorted(pos_s0 + pos_s1)

    new_candidate = []
    for win in candidate:
        arr = np.asarray(win, dtype=float)
        n_thrd, n_stress = arr.shape
        active_mask = arr.sum(axis=1) > 0
        active_rows = arr[active_mask]
        new_win = np.zeros((n_thrd, n_stress), dtype=float)
        if len(active_rows) > 0:
            idx = np.random.permutation(len(active_rows))
            for k, pos in enumerate(active_positions):
                new_win[pos] = active_rows[idx[k % len(active_rows)]]
        else:
            for k, pos in enumerate(active_positions):
                sidx = random.randrange(n_stress)
                new_win[pos, sidx] = sample_scaling_weight() / 100.0
        new_candidate.append(new_win.tolist())
    return new_candidate


def _stratify_thread_count(candidates, num_threads=None):
    """Globally stratify a batch by active thread count. Cycles K=[1..num_threads]
    across the batch so every K gets ~equal representation. Applied as a
    post-processing step on whatever the upstream proposers produced — each
    candidate's stressor mix (which stressors at what weights) is preserved
    via reuse of its own active rows; only the active-thread count is enforced.

    Disable by setting MIMESYS_STRATIFY_THREADS=0."""
    if num_threads is None:
        num_threads = NUM_THREADS
    out = []
    for i, cand in enumerate(candidates):
        K = (i % num_threads) + 1
        out.append(_apply_thread_count(cand, K))
    return out


def propose_convex_hull_and_novelty_mix(A, M, n_candidates):
    """via hull_mixed_fps_hybrid, optionally mixed with disjoint composites.

    Env vars:
      MIMESYS_COMPOSITE_FRACTION (default 0.0)  fraction of batch from composites
      MIMESYS_EXPLICIT_THREAD_FRACTION (default 0.0) fraction from
                                                   propose_candidates_explicit_threads
                                                   (deterministic K=1..N stratification)
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

    # Global thread-count stratification: cycle K=[1..NUM_THREADS] across the
    # batch so every K gets equal representation. Augmentation later permutes
    # which K positions are active. Set MIMESYS_STRATIFY_THREADS=0 to disable.
    stratify_threads = os.environ.get("MIMESYS_STRATIFY_THREADS", "1") != "0"
    if stratify_threads:
        out = _stratify_thread_count(out, num_threads=NUM_THREADS)
        # Re-shuffle so K-position isn't correlated with worker dispatch order
        random.shuffle(out)

    dropped_hull = len(raw_hull) - len(hull_fps)
    dropped_comp = max(0, len(raw_comp) - len(composites))
    print(f"  [propose] n_hull_fps={n_hull_fps}  n_composite={n_composite}  "
          f"(composite_frac={composite_frac:.2f}, "
          f"stratify_threads={stratify_threads}, dedup={do_dedup})")
    print(f"  [propose] kept hull/fps={len(hull_fps)} (dropped {dropped_hull}); "
          f"composite={len(composites)} (dropped {dropped_comp}); "
          f"total={len(out)}")

    # Tier 3.7: hard-cap idle-thread count. If MIMESYS_MIN_ACTIVE_THREADS is
    # set, drop any candidate whose curr-action has fewer than that many threads
    # with total stressor weight > MIMESYS_ACTIVE_THREAD_THRESH (default 0.1).
    min_active = int(os.environ.get("MIMESYS_MIN_ACTIVE_THREADS", "0"))
    if min_active > 0:
        active_thresh = float(os.environ.get("MIMESYS_ACTIVE_THREAD_THRESH", "0.1"))
        def _active_count(cand):
            curr = cand[-1] if len(cand) > 1 else cand[0]  # curr is the LAST window
            return sum(1 for row in curr if sum(row) > active_thresh)
        n_before = len(out)
        out = [c for c in out if _active_count(c) >= min_active]
        print(f"  [propose] tier-3.7 cap: kept {len(out)}/{n_before} candidates "
              f"(min_active_threads={min_active} at >{active_thresh})")

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
      - returns the same tuple format: (actions, avg_list, med_list, std_list)
    """
    from collections import defaultdict
    from mimesys.preprocessing.dataloader import parse_trace_file, process_trace_all
    from mimesys.preprocessing.pqos_parser import pqos_metrics_dict, PQOS_METRIC_KEYS

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
        # Merge pqos.log (per-socket LLC occupancy + memory BW + IPC + Misses)
        # into the same per-metric dict; binning happens later on (metric → list).
        # The pqos file is emitted by mimesys_benchmark's StopProfiling next to
        # stats-plan_NNN.txt as pqos-plan_NNN.log.
        pqos_path = sp.replace("/stats-plan_", "/pqos-plan_").replace(".txt", ".log")
        try:
            # libpqos in-process polling (binary v5+) produces one pqos sample
            # per slot, natively aligned with hpc — no timestamp alignment needed.
            pqos_metrics = pqos_metrics_dict(pqos_path)
        except Exception:
            pqos_metrics = {k: [] for k in PQOS_METRIC_KEYS}
        # Only attach if at least socket 0 LLC is non-empty (otherwise leave
        # zeros so downstream code doesn't crash on missing keys).
        for k in PQOS_METRIC_KEYS:
            vals = pqos_metrics.get(k, [])
            # Inject into pm with matching length to other metrics (truncate / pad to
            # the length of an existing metric so binning logic below sees same N).
            if pm:
                ref_len = max((len(v) for v in pm.values() if isinstance(v, list)), default=0)
                if len(vals) >= ref_len:
                    vals = vals[:ref_len]
                else:
                    vals = list(vals) + [0.0] * (ref_len - len(vals))
            pm[k] = vals
        with h5py.File(h5p, "r") as f:
            actions = f["execution_plan"][:].tolist()
        # A curr-only plan (prev profiling was skipped) has a single window.
        # Reconstruct the 2-window [NONE_PREV, curr] action and shift the captured
        # slots by +1 so curr lands at slot 1 (slot 0 = an all-zero prev slot),
        # matching the layout of fully-profiled 2-window plans.
        none_current = (len(actions) == 1)
        num_slots = len(actions) + 1            # captured slot groups (cycle length)
        shift = 1 if none_current else 0

        # Drop the first cycle (curr_0 + sleep_0) of measurements: curr_0
        # contains warmup-tail contamination from the kernel-init/populate
        # phase that bleeds into the first measured slot. With MIMESYS_ITERS=4
        # the kernel reports 4 currs and we keep curr_1..curr_3; with ITERS=3
        # we keep curr_1..curr_2. Toggle off with MIMESYS_KEEP_FIRST_CURR=1.
        drop_first = os.environ.get("MIMESYS_KEEP_FIRST_CURR", "0") != "1"

        # Per-group median of nonzero, per metric. Group i collects samples
        # at sample_idx % num_slots == i (samples cycle through prev / curr /
        # noop slots in the BM_Mimesys loop).
        avg_metrics = defaultdict(dict)
        med_metrics = defaultdict(dict)
        std_metrics = defaultdict(dict)
        for tm, vals in pm.items():
            if drop_first and len(vals) > num_slots:
                vals = vals[num_slots:]
            groups = defaultdict(list)
            for i, v in enumerate(vals):
                groups[i % num_slots].append(v)
            for g, gvals in groups.items():
                nz = [x for x in gvals if x != 0]
                avg_metrics[g + shift][tm] = (sum(nz) / len(nz)) if nz else 0.0
                med_metrics[g + shift][tm] = sorted(nz)[len(nz) // 2] if nz else 0.0
                std_metrics[g + shift][tm] = float(np.std(nz)) if nz else 0.0

        # Flatten as the old code did: outer-group then inner-metric order
        # (metric order = process_trace_all dict insertion order, deterministic).
        # For curr-only plans the prepended slot 0 has no entries → flattens to 0.
        out_slots = num_slots + shift
        flat_avg, flat_med, flat_std = [], [], []
        keys = list(pm.keys())
        for g in range(out_slots):
            for tm in keys:
                flat_avg.append(avg_metrics[g].get(tm, 0.0))
                flat_med.append(med_metrics[g].get(tm, 0.0))
                flat_std.append(std_metrics[g].get(tm, 0.0))

        if none_current:
            actions = as_none_current(actions[0])   # [NONE_PREV, curr]
        pairs.append((actions, flat_avg, flat_med, flat_std))
    return pairs


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
        n0 = len(A_init)
        A_init = [a for a in A_init if not _curr_is_empty(a)]   # drop no-op (all-zero curr) candidates
        if len(A_init) < n0:
            print(f"  [filter] dropped {n0 - len(A_init)} all-zero-curr candidates")
        A_init = [as_none_current(a) for a in A_init]   # prev=None (none-current)
        print(f"\n=== Round 0  (initial_candidates: {len(A_init)} plans, prev=None) ===")
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
        none_pairs, _ = _split_none_current(pairs)   # variants are training-only
        A_r, M_r, var_r = pairs_to_AM(none_pairs, max_len=M.shape[1])
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

        proposed = propose_convex_hull_and_novelty_mix(A, M, BATCH_SIZE)
        currs = [curr_window(a) for a in proposed]
        n_prop = len(currs)
        currs = [c for c in currs if not _curr_is_empty(c)]   # drop no-op (all-zero curr) candidates
        if len(currs) < n_prop:
            print(f"  [filter] dropped {n_prop - len(currs)} all-zero-curr candidates")
        # N_REP independent replicate plans per unique curr (noise-averaging
        # for pure AL; no effect when N_REP=1, which is the legacy default).
        none_plans = [as_none_current(c) for c in currs for _ in range(N_REP)]
        if N_REP > 1:
            print(f"  [N_REP] each curr replicated {N_REP}× → {len(currs)} unique → {len(none_plans)} plans")

        # K prev-variants per curr: prev = diverse none-current sample from the pool
        # (spanning the metric space → high LLC / mem-BW prevs). Profiled as real
        # 2-window plans alongside the none-current curr-only plans.
        variants, prev_foot = build_prev_variants(currs, A, M, K_PREV_VARIANTS)
        if variants:
            llcs = [f[2] for f in prev_foot]; bws = [f[3] for f in prev_foot]
            print(f"  + {len(variants)} prev-variants ({K_PREV_VARIANTS}/curr × {PREV_N_REP} reps, "
                  f"high-LLC top {PREV_HIGH_LLC_FRAC:.0%}, {len(prev_foot)}-prev shortlist); "
                  f"shortlist LLC {min(llcs):.0f}-{max(llcs):.0f} MB/s, BW {min(bws):.1f}-{max(bws):.1f} %")
        # Interleave so each curr's (NONE × N_REP) + (K_PREV × N_REP) plans are
        # contiguous in the batch. write_actions_to_execution_plans then slices
        # the batch sequentially across machines, so all plans for a given curr
        # land on the same host — kills the host-id confound in prev-impact
        # measurement.
        per_curr_variants = K_PREV_VARIANTS * PREV_N_REP
        batch = []
        for i in range(len(currs)):
            for j in range(N_REP):
                batch.append(none_plans[i * N_REP + j])
            batch.extend(variants[i * per_curr_variants : (i + 1) * per_curr_variants])
        print(f"  [interleave] {len(currs)} currs × ({N_REP} none + {per_curr_variants} prev) "
              f"= {len(batch)} plans, contiguous-per-curr (same-host NONE↔PREV)")

        dest = os.path.join(OUTPUT_PATH, f"round_{r}")
        write_actions_to_execution_plans(batch, dest, PROFILING_MACHINES)

        # Dispatch + extract zips on workers; skip the buggy in-line parse.
        asyncio.run(profiler.profile_actions(dest, skip_parsing=True))
        # Clean parse from on-disk raw stats. Only the none-current samples (zero
        # prev) feed the active-learning pool; prev-variants are training-only.
        pairs = load_round(profiler, r)
        none_pairs, var_pairs = _split_none_current(pairs)
        A_r, M_r, var_r = pairs_to_AM(none_pairs, max_len=M.shape[1])
        n_valid = 0 if M_r is None else len(A_r)
        print(f"  Valid: {n_valid} none-current + {len(var_pairs)} prev-variants  (of {len(pairs)} plans)")

        if M_r is None:
            print("  No valid none-current results, skipping round")
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
