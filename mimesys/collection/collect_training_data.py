"""AL collection: hull-gap-fill selection of N_SELECT plans per round out of
N_CANDIDATES, scored by 28 single-output RF surrogates.

Round 0 is a one-hot kernel sweep (seed_round0); metric ranges come from the
SEED_FROM collection. Rounds 1..N_ROUNDS_NEW:
  1. Generate candidates: sample_plans (odd rounds) or thread-disjoint unions
     of reference pairs (even rounds)
  2. Fit 28 single-output RandomForestRegressors on action→metric from all
     completed rounds
  3. Predict candidate metrics; select N_SELECT via hull-gap filling
     (K-stratified / multi-pair / single-pair, env-selected) with FPS-novel
     fallback
  4. Distribute to the configured worker hosts and profile

Output:    $OUT_DIR/round_N/{chunk_*}/

Usage:
  cd mimesys/collection
  [OUT_DIR=...] [SEED_FROM=...] [N_ROUNDS_NEW=3] python collect_training_data.py

Completed rounds are detected and skipped, so re-running resumes an
interrupted collection.
"""
import sys, os, time, pickle, json, asyncio, subprocess, glob, shutil
import random
from pathlib import Path
import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "worker_scripts"))
import config as worker_config

from mimesys.collection.profiling_server import InitializeRequest, Profiler
from mimesys.schema.machine import Machine
from mimesys.preprocessing.parsers import parse_trace_file, process_trace_all
from mimesys.collection.profiling_server import _merge_pqos_into_metrics
from mimesys.preprocessing.pqos_parser import pqos_metrics_dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# ───── config ─────
OUT_DIR        = os.environ.get("OUT_DIR", os.path.expanduser("~/mimesys_training_data/al_singleRF_hullfill_v1"))
SEED_FROM      = os.environ.get("SEED_FROM", os.path.expanduser("~/mimesys_training_data/training_data_round1_1sec"))  # source of metrics_range_dict.pkl
N_ROUNDS_NEW   = int(os.environ.get("N_ROUNDS_NEW", "3"))    # round_1..N
N_CANDIDATES   = int(os.environ.get("N_CANDIDATES", "512"))
N_SELECT       = int(os.environ.get("N_SELECT", "128"))
NUM_STRESSORS  = 20
NUM_THREADS    = 20
HOSTS          = list(worker_config.HOSTNAMES)
H              = len(HOSTS)
RANGE_PKL      = f"{SEED_FROM}/metrics_range_dict.pkl"

# sampler bounds
KT_MIN, KT_MAX  = 1, 20
KS_MIN, KS_MAX  = 1, 10
W_MIN,  W_MAX   = 0.2, 1.0
POS_ALPHA       = 0.0

# scoring
RF_N_TREES   = 80
RF_MAX_DEPTH = 12
LAM_UNCERT   = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

# 1-sec slot windows on the workers; the Profiler reads these env vars and
# prepends them to the remote bench command. (The 2-sec default overshoots
# under multi-stressor mixes.)
os.environ["MIMESYS_SLOT_US"] = "1000000"  # 1 s
os.environ["MIMESYS_ITERS"]   = "4"
os.environ["MIMESYS_SLEEP"]   = "1"

# ───── prev-action variant sampling config (see build_prev_variants) ─────
# Per selected curr, optionally emit K prev-variant plans [prev, curr] with
# prev drawn from the reference pool (rounds 0..r-1, none-current only).
# PREV_VARIANTS=0 (default) disables → curr-only rounds.
# PREV_SHORTLIST_MULT — FPS shortlist size = K * mult; larger spreads prevs
#   further across the (LLC, BW, IO) space.
K_PREV_VARIANTS     = int(os.environ.get("PREV_VARIANTS",       "0"))   # 0 disables
PREV_SHORTLIST_MULT = int(os.environ.get("PREV_SHORTLIST_MULT", "10"))
PREV_HIGH_RESOURCE_FRAC  = float(os.environ.get("PREV_HIGH_RESOURCE_FRAC","0.5"))


# ───── plan generation / writing helpers ─────
# All-zero prev window ("no prior action"). write_actions_to_execution_plans
# drops it, so none-current plans [NONE_PREV, curr] are written curr-only and
# the worker skips profiling the empty prev; readers reconstruct the 2-window
# form (see preprocessing/dataloader.py).
NUM_ACTIONS = int(os.environ.get("MIMESYS_NUM_ACTIONS", "19"))
NONE_PREV = [[0.0] * NUM_ACTIONS for _ in range(NUM_THREADS)]


def initial_candidates(bounds, n_candidates, num_max_threads=20):
    """ Generate initial candidates as one-hot vectors within the given bounds.

    Two modes selectable via env var ``MIMESYS_INITIAL_OLD_STYLE`` (default 0):
      "0" — sweep mode: num_threads ∈ [1, candidate_max_threads] × weight grid.
            Produces ~2000 sparse-dominant seeds per round 0.
      "1" — only num_threads=candidate_max and weight=1.0. Produces ~70 dense
            seeds per round 0, matching the test per-core CPU profile.
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

        # Every stressor (including the IO-bound ones) gets the same
        # num_threads × weight grid, with a single shuffled thread layout per
        # combination — no per-stressor thread cap, so the per-core CPU%
        # distribution is shaped by the workload itself.
        candidate_max_threads = num_max_threads

        if old_style:
            # Single iter at max threads, single weight (1.0)
            num_threads_range = [candidate_max_threads]
            weight_range = [1.0]
        else:
            num_threads_range = range(1, candidate_max_threads + 1)
            # MIMESYS_INITIAL_WEIGHT_RANGE (comma-separated, e.g. "1.0")
            # overrides the default weight grid.
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
        # none-current: prev = NONE_PREV
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
            # curr-only and skip profiling the empty prev. Never drop the curr
            # (last) window or emit an empty plan: an empty H5 is not 3D and
            # segfaults the benchmark ("Dataset is not 3D").
            windows = list(action)
            while len(windows) > 1 and all(all(v == 0.0 for v in t) for t in windows[0]):
                windows.pop(0)
            write_to_hdf5(windows, file_path)


# ───── seed round_0 from existing collection ─────
def seed_round0():
    """Round 0: one-hot sweep of every kernel × num_threads × weight via
    initial_candidates. For 20 stressors × 20 thread
    counts × 4 weights (0.25/0.5/0.75/1.0), that yields ~1600 sparse-dominant
    plans covering each kernel in isolation across the K∈[1,20] grid.

    Returns (N, S, T) float32 array ready for write_round, or None if the
    round is already collected.
    """
    dst = f"{OUT_DIR}/round_0"
    if os.path.exists(dst):
        n_stats = len(glob.glob(f"{dst}/chunk_*/results/stats-plan_*.txt"))
        if n_stats > 0:
            print(f"[al] round_0 already collected ({n_stats} stats) — skipping")
            return None
    print(f"[al] round_0: generating one-hot sweep via initial_candidates")
    # `bounds` is only used for its length in initial_candidates (k = num_actions).
    bounds = [(0.0, 1.0)] * NUM_STRESSORS
    wrapped = initial_candidates(bounds, n_candidates=NUM_STRESSORS,
                                  num_max_threads=NUM_THREADS)
    # Each entry is [NONE_PREV, curr] where curr is list[T=20][S=20].
    # write_round expects (K, S, T); convert via transpose.
    plans = np.array([np.array(p[1]).T for p in wrapped], dtype=np.float32)
    plans = _cap_sumw_per_thread(plans, cap=1.0)
    print(f"[al] round_0: {plans.shape[0]} plans (shape {plans.shape})")
    return plans


# ───── load metric ranges (for normalization) ─────
with open(RANGE_PKL, "rb") as f:
    RANGES = pickle.load(f)
METRIC_KEYS = sorted(RANGES.keys())
M_MIN  = np.array([RANGES[k][0] for k in METRIC_KEYS], dtype=np.float32)
M_MAX  = np.array([RANGES[k][1] for k in METRIC_KEYS], dtype=np.float32)
M_RNG  = np.where(M_MAX - M_MIN > 0, M_MAX - M_MIN, 1.0)


# ───── parse stats file into (action, metric) pair ─────
def _load_round_pairs(round_dir):
    """For each plan_NNN.h5 with paired stats-plan_NNN.txt in round_dir, return
    (action (S, T), raw metric (len(METRIC_KEYS),), is_variant) triples."""
    pairs = []
    chunks = sorted(glob.glob(f"{round_dir}/chunk_*"))
    for ch in chunks:
        plans_dir = f"{ch}/plans"
        results_dir = f"{ch}/results"
        if not os.path.isdir(plans_dir) or not os.path.isdir(results_dir):
            continue
        for h5 in sorted(glob.glob(f"{plans_dir}/plan_*.h5")):
            idx = os.path.basename(h5).replace("plan_", "").replace(".h5", "")
            stats_p = f"{results_dir}/stats-plan_{idx}.txt"
            if not os.path.exists(stats_p): continue
            try:
                import h5py
                with h5py.File(h5) as f:
                    ep = f["execution_plan"][:]               # (W, T, S) or (W, S, T)
                w = ep[-1]  # last window, (T, S) or (S, T)
                # Normalize to (S, T). Legacy plans with NUM_STRESSORS-1
                # stressors lack the newest kernel: zero-pad it.
                if w.shape == (NUM_THREADS, NUM_STRESSORS):
                    action = w.T.astype(np.float32)            # (S, T)
                elif w.shape == (NUM_THREADS, NUM_STRESSORS - 1):
                    pad = np.zeros((NUM_THREADS, 1), dtype=np.float32)
                    action = np.concatenate([w, pad], axis=1).T.astype(np.float32)
                elif w.shape == (NUM_STRESSORS - 1, NUM_THREADS):
                    pad = np.zeros((1, NUM_THREADS), dtype=np.float32)
                    action = np.concatenate([w, pad], axis=0).astype(np.float32)
                else:
                    action = w.astype(np.float32)              # already (S, T)
                _, parsed = parse_trace_file(stats_p)
                mets = process_trace_all(parsed)
                if mets is None: continue
                # Drop samples whose HPC capture is truncated relative to the
                # pqos slots — their pqos metrics distort the hull.
                hpc_len = len(next(iter(mets.values())))
                pqos_p = stats_p.replace("/stats-", "/pqos-").replace(".txt", ".log")
                if os.path.exists(pqos_p):
                    try:
                        pq_d = pqos_metrics_dict(pqos_p)
                        pq_len = len(pq_d.get("pqos_ipc", []))
                        if pq_len > 0 and pq_len != hpc_len:
                            continue
                    except Exception:
                        pass
                mets = _merge_pqos_into_metrics(mets, stats_p)
                # Slot cycle length = n_windows + 1 (each window profiled + one sleep).
                # For 1-window (none-current) plans: cycle=[curr, sleep], curr at idx 0.
                # For 2-window (prev-variant) plans: cycle=[prev, curr, sleep], curr at idx 1.
                n_windows     = int(ep.shape[0])
                num_slots     = n_windows + 1
                curr_slot_idx = n_windows - 1
                is_variant    = (n_windows > 1)
                metric_vec = []
                for k in METRIC_KEYS:
                    vals = mets.get(k, [])
                    curr = vals[curr_slot_idx::num_slots] if len(vals) > curr_slot_idx else vals
                    metric_vec.append(float(np.median(curr[1:])) if len(curr) > 1 else 0.0)
                pairs.append((action, np.array(metric_vec, dtype=np.float32), is_variant))
            except Exception as e:
                print(f"  [parse fail] {h5}: {e}")
    return pairs


def load_ref_through_round(max_round):
    """Load NONE-CURRENT (action, raw_metric) pairs from round_0 .. round_max_round.

    Prev-variant plans (n_windows>1 h5 files) are skipped for the AL reference
    pool — their curr metrics carry prev-carryover and would poison hull/FPS
    scoring. They remain on disk for the training pipeline.
    """
    all_a, all_m = [], []
    for r in range(max_round + 1):
        rd = f"{OUT_DIR}/round_{r}"
        if not os.path.exists(rd): continue
        pairs = _load_round_pairs(rd)
        n_none = n_var = 0
        for a, m, is_variant in pairs:
            if is_variant:
                n_var += 1
                continue
            all_a.append(a); all_m.append(m); n_none += 1
        tag = f" ({n_var} prev-variants skipped for AL ref)" if n_var else ""
        print(f"  [load] round_{r}: {n_none} none-current samples{tag}")
    A = np.stack(all_a) if all_a else np.zeros((0, NUM_STRESSORS, NUM_THREADS), np.float32)
    M = np.stack(all_m) if all_m else np.zeros((0, len(METRIC_KEYS)), np.float32)
    return A, M


# ───── candidate sampler (same as collection_viz_server.sample_plans) ─────
# Stressor classification (0-indexed against mimesys_actions.txt):
#   CPU: 0,1,2,3,4,5,6,7  + 15,16,17,18 (DIRECTMEMSET/SIMD use RAM, not disk)
#   IO:  8,9,10,11,12,13,14 (HddRead / HddWrite)
# Each active thread is HOMOGENEOUS — either pure CPU or pure IO; IO stressors
# block on iowait and crush a mixed thread's user-CPU.
CPU_STRESSORS = [0, 1, 2, 3, 4, 5, 6, 7, 15, 17, 18, 19]
# s16 (DIRECTMEMSET_32MB_WeightScaled_50ms) lives here so the sampler forces
# K_s=1: its built-in 50ms sleep budget composes poorly with full-duty CPU
# stressors.
IO_STRESSORS  = [8, 9, 10, 11, 12, 13, 14, 16]

def sample_plans(n_plans, seed):
    """Per-thread homogeneous CPU/IO mode. Per-plan IO probability ~ Uniform(0,1)
    spans the full CPU↔IO Pareto frontier so AL can find both corners.

    Within a thread: budget B ∈ [W_MIN, 1.0] split via Dirichlet across K_s
    stressors. Per-thread sum_w ≤ 1.0. IO threads use K_s=1 (IO doesn't compose);
    CPU threads use K_s ∈ [KS_MIN, KS_MAX]."""
    rng = np.random.default_rng(seed)
    plans = np.zeros((n_plans, NUM_STRESSORS, NUM_THREADS), dtype=np.float32)
    pos_logits = -POS_ALPHA * np.arange(NUM_THREADS, dtype=np.float32)
    pos_prob = np.exp(pos_logits - pos_logits.max()); pos_prob /= pos_prob.sum()
    for n in range(n_plans):
        Kt = int(rng.integers(KT_MIN, KT_MAX + 1))
        active = rng.choice(NUM_THREADS, size=Kt, replace=False, p=pos_prob)
        p_io = float(rng.uniform(0.0, 1.0))   # per-plan IO-thread probability
        for t in active:
            is_io_thread = rng.random() < p_io
            pool = IO_STRESSORS if is_io_thread else CPU_STRESSORS
            Ks = 1 if is_io_thread else int(rng.integers(KS_MIN, KS_MAX + 1))
            Ks = min(Ks, len(pool))
            chosen = rng.choice(pool, size=Ks, replace=False)
            B = float(rng.uniform(W_MIN, 1.0))         # per-thread budget
            w = rng.dirichlet(np.ones(Ks)) * B          # sum = B
            plans[n, chosen, t] = w.astype(np.float32)
    return np.clip(plans, 0.0, 1.0)


def _cap_sumw_per_thread(plans, cap=1.0):
    """Scale each thread so sum_of_weights ≤ cap. Returns a copy."""
    out = plans.copy().astype(np.float32)
    sums = out.sum(axis=1, keepdims=True)              # (N, 1, T)
    scale = np.minimum(1.0, cap / np.maximum(sums, 1e-9))
    return out * scale


# 2-D metric projections targeted for hull-gap filling each round
HULL_PAIRS = [
    ("cpu_mean",              "io_read"),
    ("cpu_mean",              "io_write"),
    ("io_read",               "io_write"),
    ("memory_bandwidth_read", "memory_bandwidth_write"),
    ("l3_cache_usage",        "memory_bandwidth_read"),
    ("l3_cache_usage",        "memory_bandwidth_write"),
    ("cpu_mean",              "l3_cache_usage"),
    ("cpu_mean",              "memory_bandwidth_read"),
    ("cpu_mean",              "memory_bandwidth_write"),
    ("cpu_mean",              "pqos_ipc"),
    ("pqos_llc_kb",           "pqos_misses"),
]


def _axis_vals(name, arr):
    if name == "cpu_mean":   return arr[:, :20].mean(axis=1)
    if name == "K_active":   return (arr[:, :20] > 50).sum(axis=1).astype(np.float32)
    if name in METRIC_KEYS:  return arr[:, METRIC_KEYS.index(name)]
    raise ValueError(name)


def generate_union_candidates(ref_action, ref_metric_raw, pairs,
                              grid_n, empty_pct, n_per_cell=3, max_K=20,
                              seed=0):
    """For each empty (x,y) hull cell, find the ref-pair (A,B) whose summed
    metrics (M[A] + M[B]) are closest to the cell center AND whose summed
    active-thread count K_A + K_B ≤ max_K. Emit the thread-disjoint UNION
    of A and B (B's stressor rows remapped onto A's idle thread slots) as
    a candidate.

    Summed metrics are only an additive approximation (contention makes real
    unions sub-additive); no correction is applied here — the round's RF
    surrogate rescores the unions in the selector.
    """
    from scipy.spatial import ConvexHull, Delaunay
    rng = np.random.default_rng(seed)

    # ref_action is (N, S, T); a thread is active if its summed weight > 0.05.
    ref_active_mask = (ref_action.sum(axis=1) > 0.05)        # (N, T)
    K_active        = ref_active_mask.sum(axis=1).astype(np.int32)  # (N,)

    out = []
    for ax, ay in pairs:
        rx = _axis_vals(ax, ref_metric_raw)
        ry = _axis_vals(ay, ref_metric_raw)
        # Same empty-cell logic as `_empty_cells_for_pair`, inlined because
        # the cell centers are needed here.
        pts = np.stack([rx, ry], axis=1)
        try:
            ConvexHull(pts); delaunay = Delaunay(pts)
        except Exception:
            delaunay = None
        x_lo, x_hi = float(np.percentile(rx, 1)), float(np.percentile(rx, 99))
        y_lo, y_hi = float(np.percentile(ry, 1)), float(np.percentile(ry, 99))
        if x_lo == x_hi or y_lo == y_hi:
            continue
        gx = np.linspace(x_lo, x_hi, grid_n + 1)
        gy = np.linspace(y_lo, y_hi, grid_n + 1)
        cell_count, _, _ = np.histogram2d(rx, ry, bins=[gx, gy])
        cell_count = cell_count.T
        centers_x = (gx[:-1] + gx[1:]) / 2
        centers_y = (gy[:-1] + gy[1:]) / 2
        CX, CY = np.meshgrid(centers_x, centers_y)
        centers = np.stack([CX.ravel(), CY.ravel()], axis=1)
        in_hull = ((delaunay.find_simplex(centers) >= 0)
                   .reshape(grid_n, grid_n)
                   if delaunay is not None
                   else np.ones_like(cell_count, dtype=bool))
        in_hull_counts = cell_count[in_hull]
        if not len(in_hull_counts):
            continue
        thresh = float(np.percentile(in_hull_counts, empty_pct))
        empty_mask = in_hull & (cell_count <= thresh)
        if not empty_mask.any():
            continue
        x_scale = max(x_hi - x_lo, 1e-9)
        y_scale = max(y_hi - y_lo, 1e-9)

        # Vectorized scoring over all (i<j) ref pairs.
        N = ref_action.shape[0]
        K_sum = K_active[:, None] + K_active[None, :]        # (N, N)
        budget_ok = K_sum <= max_K
        triu = np.triu(np.ones((N, N), dtype=bool), k=1)
        valid = budget_ok & triu

        sum_x = rx[:, None] + rx[None, :]                    # (N, N)
        sum_y = ry[:, None] + ry[None, :]

        n_emit = 0
        for i in range(grid_n):
            for j in range(grid_n):
                if not empty_mask[i, j]: continue
                cx_c = centers_x[j]; cy_c = centers_y[i]
                d = (((sum_x - cx_c) / x_scale) ** 2 +
                     ((sum_y - cy_c) / y_scale) ** 2)
                d[~valid] = np.inf
                flat = d.ravel()
                if not np.isfinite(flat).any(): continue
                k_top = min(n_per_cell, int(np.isfinite(flat).sum()))
                top_idx = np.argpartition(flat, k_top - 1)[:k_top]
                # Sort top-k by distance so we keep best first.
                top_idx = top_idx[np.argsort(flat[top_idx])]
                for fi in top_idx:
                    ia, ib = int(fi // N), int(fi % N)
                    # Build thread-disjoint union: A's rows kept; B's active
                    # threads remapped to A's idle thread slots.
                    active_a = np.where(ref_active_mask[ia])[0]
                    active_b = np.where(ref_active_mask[ib])[0]
                    idle_a   = [t for t in range(NUM_THREADS) if t not in active_a]
                    if len(active_b) > len(idle_a):
                        continue
                    union = ref_action[ia].copy()
                    rng.shuffle(idle_a)
                    for k_b, b_thr in enumerate(active_b):
                        union[:, idle_a[k_b]] = ref_action[ib][:, b_thr]
                    out.append(union.astype(np.float32))
                    n_emit += 1
        print(f"    {ax:>26} × {ay:<26}  union emitted={n_emit}")
    if not out:
        S = ref_action.shape
        return np.zeros((0, S[1], S[2]), dtype=np.float32)
    return np.stack(out)


def _empty_cells_for_pair(rx, ry, cx, cy, grid_n, empty_pct):
    """Return (empty_mask, gx, gy, cand_row, cand_col, ref_in_hull) for one 2-D pair."""
    from scipy.spatial import ConvexHull, Delaunay
    pts = np.stack([rx, ry], axis=1)
    delaunay = None
    try:
        ConvexHull(pts); delaunay = Delaunay(pts)
    except Exception: pass
    x_lo, x_hi = float(np.percentile(rx, 1)), float(np.percentile(rx, 99))
    y_lo, y_hi = float(np.percentile(ry, 1)), float(np.percentile(ry, 99))
    if x_lo == x_hi or y_lo == y_hi:
        return None
    gx = np.linspace(x_lo, x_hi, grid_n + 1)
    gy = np.linspace(y_lo, y_hi, grid_n + 1)
    cell_count, _, _ = np.histogram2d(rx, ry, bins=[gx, gy])
    cell_count = cell_count.T
    centers_x = (gx[:-1] + gx[1:]) / 2; centers_y = (gy[:-1] + gy[1:]) / 2
    CX, CY = np.meshgrid(centers_x, centers_y)
    centers = np.stack([CX.ravel(), CY.ravel()], axis=1)
    in_hull = (delaunay.find_simplex(centers) >= 0).reshape(grid_n, grid_n) \
              if delaunay is not None else np.ones_like(cell_count, dtype=bool)
    in_hull_counts = cell_count[in_hull]
    thresh = float(np.percentile(in_hull_counts, empty_pct)) if len(in_hull_counts) else 0
    empty_mask = in_hull & (cell_count <= thresh)
    cand_col = np.clip(np.searchsorted(gx, cx) - 1, 0, grid_n - 1)
    cand_row = np.clip(np.searchsorted(gy, cy) - 1, 0, grid_n - 1)
    return empty_mask, cand_row, cand_col


def select_top_K_hullfill_multi(candidates_action, ref_action, ref_metric_raw, K,
                                 pairs=None, grid_n=20, empty_pct=20.0):
    """Multi-pair hull-gap-fill. Budget split uniformly across pairs; for each
    pair we pick the (budget) candidates landing inside its empty interior cells.
    De-duped across pairs (a candidate covering empty cells in multiple pairs is
    still kept once). Any unused budget falls back to FPS-novel."""
    if pairs is None: pairs = HULL_PAIRS

    # ---- 0. RFs ----
    ref_metric_norm = ((ref_metric_raw - M_MIN) / M_RNG) * 2.0 - 1.0
    Aref_flat = ref_action.reshape(ref_action.shape[0], -1)
    sx = StandardScaler().fit(Aref_flat); sy = StandardScaler().fit(ref_metric_norm)
    Xfit = sx.transform(Aref_flat);  Yfit = sy.transform(ref_metric_norm)
    t0 = time.time()
    rfs = []
    for i in range(len(METRIC_KEYS)):
        rf = RandomForestRegressor(n_estimators=RF_N_TREES, n_jobs=-1,
                                    random_state=0, max_depth=RF_MAX_DEPTH)
        rf.fit(Xfit, Yfit[:, i])
        rfs.append(rf)
    print(f"  [rf] fit {len(rfs)} single-output RFs in {time.time()-t0:.1f}s")

    # ---- 1. Predict candidates to raw metric ----
    Acan_flat = candidates_action.reshape(candidates_action.shape[0], -1)
    Xcan = sx.transform(Acan_flat)
    pred_scaled = np.stack([rf.predict(Xcan) for rf in rfs], axis=1)
    pred_norm   = sy.inverse_transform(pred_scaled)
    raw_pred    = np.zeros_like(pred_norm)
    for i, k in enumerate(METRIC_KEYS):
        lo, hi = M_MIN[i], M_MAX[i]
        raw_pred[:, i] = ((pred_norm[:, i] + 1.0) * 0.5) * (hi - lo) + lo

    # ---- 2. Per-pair empty cells + candidate buckets ----
    rng = np.random.default_rng(0)
    per_pair_budget = max(1, K // len(pairs))
    leftover_budget = K - per_pair_budget * len(pairs)
    print(f"  [hullfill-multi] {len(pairs)} pairs, budget={per_pair_budget}/pair (+ {leftover_budget} spillover)")

    selected_set = set()
    pair_stats = []
    for (ax, ay) in pairs:
        rx = _axis_vals(ax, ref_metric_raw); ry = _axis_vals(ay, ref_metric_raw)
        cx = _axis_vals(ax, raw_pred);       cy = _axis_vals(ay, raw_pred)
        out = _empty_cells_for_pair(rx, ry, cx, cy, grid_n, empty_pct)
        if out is None:
            pair_stats.append((ax, ay, 0, 0, 0)); continue
        empty_mask, cand_row, cand_col = out
        n_empty = int(empty_mask.sum())

        # Bucket candidates by empty cell, preferring those not yet selected
        cell_to_cand = {}
        for i in range(len(cx)):
            r, c = int(cand_row[i]), int(cand_col[i])
            if empty_mask[r, c] and i not in selected_set:
                cell_to_cand.setdefault((r, c), []).append(i)

        picked_for_pair = 0
        for (r, c), idxs in cell_to_cand.items():
            if picked_for_pair >= per_pair_budget: break
            pick = int(idxs[rng.integers(0, len(idxs))])
            selected_set.add(pick); picked_for_pair += 1
        pair_stats.append((ax, ay, n_empty, len(cell_to_cand), picked_for_pair))

    print(f"  [hullfill-multi] per-pair (empty, candidates-available, picked):")
    for ax, ay, n_empty, n_buckets, picked in pair_stats:
        print(f"    {ax:>26} × {ay:<26}  empty={n_empty:>4}  buckets={n_buckets:>4}  picked={picked:>3}")
    print(f"  [hullfill-multi] total picked={len(selected_set)} / {K}")

    # ---- 3. Fill leftover budget with FPS-novel in standardized full metric space ----
    if len(selected_set) < K:
        ref_S = (ref_metric_norm + 1.0) * 0.5
        pred_S = np.clip((pred_norm + 1.0) * 0.5, 0.0, 1.0)
        # standardize per dim by IQR
        q25 = np.percentile(ref_S, 25, axis=0); q75 = np.percentile(ref_S, 75, axis=0)
        iqr = np.maximum(q75 - q25, 1e-3)
        ref_Z = ref_S / iqr; pred_Z = pred_S / iqr
        # distance to existing + already-selected
        anchors = np.vstack([ref_Z] + ([pred_Z[list(selected_set)]] if selected_set else []))
        chunk = 1024
        dist = np.full(len(pred_Z), np.inf)
        for s in range(0, len(pred_Z), chunk):
            e = min(s + chunk, len(pred_Z))
            d = np.linalg.norm(pred_Z[s:e, None, :] - anchors[None, :, :], axis=-1)
            dist[s:e] = d.min(axis=1)
        # Greedy FPS
        for j in np.argsort(-dist):
            if len(selected_set) >= K: break
            if int(j) in selected_set: continue
            selected_set.add(int(j))
        print(f"  [hullfill-multi] filled to K={K} via FPS-novel leftover")

    return np.array(sorted(selected_set))


def select_top_K_hullfill_kstrat(candidates_action, ref_action, ref_metric_raw, K,
                                  pairs=None, grid_n=20, empty_pct=20.0,
                                  K_min=1, K_max=20):
    """K-stratified variant: allocate K/(K_max-K_min+1) budget per K_active
    bucket (K=K_min..K_max). Within each bucket, run per-pair empty-cell hull
    targeting, falling back to FPS-novel inside the bucket. Final spillover
    falls back to global FPS-novel. K_min=1 by default to exclude pure noop
    plans (K=0)."""
    if pairs is None: pairs = HULL_PAIRS

    ref_metric_norm = ((ref_metric_raw - M_MIN) / M_RNG) * 2.0 - 1.0
    Aref_flat = ref_action.reshape(ref_action.shape[0], -1)
    sx = StandardScaler().fit(Aref_flat); sy = StandardScaler().fit(ref_metric_norm)
    Xfit = sx.transform(Aref_flat);  Yfit = sy.transform(ref_metric_norm)
    t0 = time.time()
    rfs = []
    for i in range(len(METRIC_KEYS)):
        rf = RandomForestRegressor(n_estimators=RF_N_TREES, n_jobs=-1,
                                    random_state=0, max_depth=RF_MAX_DEPTH)
        rf.fit(Xfit, Yfit[:, i])
        rfs.append(rf)
    print(f"  [rf] fit {len(rfs)} single-output RFs in {time.time()-t0:.1f}s")

    Acan_flat = candidates_action.reshape(candidates_action.shape[0], -1)
    Xcan = sx.transform(Acan_flat)
    pred_scaled = np.stack([rf.predict(Xcan) for rf in rfs], axis=1)
    pred_norm   = sy.inverse_transform(pred_scaled)
    raw_pred    = np.zeros_like(pred_norm)
    for i, k in enumerate(METRIC_KEYS):
        lo, hi = M_MIN[i], M_MAX[i]
        raw_pred[:, i] = ((pred_norm[:, i] + 1.0) * 0.5) * (hi - lo) + lo

    # K_active per candidate
    K_per_cand = (candidates_action.sum(axis=1) > 0.5).sum(axis=1)

    n_K_buckets = K_max - K_min + 1
    base_budget = K // n_K_buckets
    spillover = K - base_budget * n_K_buckets

    rng = np.random.default_rng(0)
    selected_set = set()
    bucket_results = []

    # REF axes, reused across K buckets
    ref_axes = {(ax, ay): (_axis_vals(ax, ref_metric_raw), _axis_vals(ay, ref_metric_raw))
                for ax, ay in pairs}
    # FPS-novel anchors prep
    ref_S = (ref_metric_norm + 1.0) * 0.5
    pred_S = np.clip((pred_norm + 1.0) * 0.5, 0.0, 1.0)
    q25 = np.percentile(ref_S, 25, axis=0); q75 = np.percentile(ref_S, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-3)
    ref_Z = ref_S / iqr
    pred_Z = pred_S / iqr

    for bi, K_val in enumerate(range(K_min, K_max + 1)):
        bucket_idx = np.where(K_per_cand == K_val)[0]
        budget = base_budget + (1 if bi < spillover else 0)
        if len(bucket_idx) == 0 or budget == 0:
            bucket_results.append((K_val, len(bucket_idx), budget, 0)); continue

        bucket_picked = 0

        # Per-pair hull targeting on this bucket's candidates
        for ax, ay in pairs:
            if bucket_picked >= budget: break
            rx, ry = ref_axes[(ax, ay)]
            cx = _axis_vals(ax, raw_pred[bucket_idx])
            cy = _axis_vals(ay, raw_pred[bucket_idx])
            out = _empty_cells_for_pair(rx, ry, cx, cy, grid_n, empty_pct)
            if out is None: continue
            empty_mask, cand_row, cand_col = out

            cell_to_cand = {}
            for local_i in range(len(bucket_idx)):
                global_i = int(bucket_idx[local_i])
                if global_i in selected_set: continue
                r, c = int(cand_row[local_i]), int(cand_col[local_i])
                if empty_mask[r, c]:
                    cell_to_cand.setdefault((r, c), []).append(global_i)
            for (r, c), idxs in cell_to_cand.items():
                if bucket_picked >= budget: break
                pick = int(idxs[rng.integers(0, len(idxs))])
                selected_set.add(pick); bucket_picked += 1

        # FPS-novel within bucket for leftover
        if bucket_picked < budget:
            bucket_pred_Z = pred_Z[bucket_idx]
            # already-selected anchors from REF + globally selected
            sel_global = [i for i in selected_set]
            anchor_extra = pred_Z[sel_global] if sel_global else np.empty((0, pred_Z.shape[1]))
            anchors = np.vstack([ref_Z, anchor_extra])
            dist = np.full(len(bucket_idx), np.inf)
            for s in range(0, len(bucket_idx), 1024):
                e = min(s + 1024, len(bucket_idx))
                d = np.linalg.norm(bucket_pred_Z[s:e, None, :] - anchors[None, :, :], axis=-1)
                dist[s:e] = d.min(axis=1)
            for j in np.argsort(-dist):
                if bucket_picked >= budget: break
                global_i = int(bucket_idx[j])
                if global_i in selected_set: continue
                selected_set.add(global_i); bucket_picked += 1

        bucket_results.append((K_val, len(bucket_idx), budget, bucket_picked))

    print(f"  [k-strat] per-K-bucket (n_avail, budget, picked):")
    for K_val, n_avail, budget, picked in bucket_results:
        print(f"    K={K_val:>2}  avail={n_avail:>5}  budget={budget:>2}  picked={picked:>2}")
    print(f"  [k-strat] total picked={len(selected_set)} / {K}")

    # Global FPS-novel if still under K (some buckets had < budget candidates)
    if len(selected_set) < K:
        sel_global = list(selected_set)
        anchors = np.vstack([ref_Z, pred_Z[sel_global]] if sel_global else [ref_Z])
        dist = np.full(len(pred_Z), np.inf)
        for s in range(0, len(pred_Z), 1024):
            e = min(s + 1024, len(pred_Z))
            d = np.linalg.norm(pred_Z[s:e, None, :] - anchors[None, :, :], axis=-1)
            dist[s:e] = d.min(axis=1)
        for j in np.argsort(-dist):
            if len(selected_set) >= K: break
            if int(j) in selected_set: continue
            selected_set.add(int(j))
        print(f"  [k-strat] filled to K={K} via global FPS-novel leftover")

    return np.array(sorted(selected_set))


# ───── targeted gap-filling on a 2-D hull projection ─────
def select_top_K_hullfill(candidates_action, ref_action, ref_metric_raw, K,
                          x_metric="cpu_mean", y_metric="io_read",
                          grid_n=20, empty_pct=20.0):
    """Returns indices into candidates_action selected by hull-gap fill.

    Steps:
      1. Train 28 single-output RFs.
      2. Forward-predict candidate metrics, project to (x_metric, y_metric).
      3. Compute convex hull of REF in that 2-D projection.
      4. Grid the bbox into grid_n × grid_n cells; find empty cells INSIDE hull
         (cells with REF count ≤ bottom empty_pct percentile of in-hull cells).
      5. For each empty cell, greedily pick ONE candidate that lands inside.
      6. If fewer than K cells covered, fill remaining budget with FPS-novel.
    """
    from scipy.spatial import ConvexHull, Delaunay

    ref_metric_norm = ((ref_metric_raw - M_MIN) / M_RNG) * 2.0 - 1.0
    Aref_flat = ref_action.reshape(ref_action.shape[0], -1)
    sx = StandardScaler().fit(Aref_flat); sy = StandardScaler().fit(ref_metric_norm)
    Xfit = sx.transform(Aref_flat);  Yfit = sy.transform(ref_metric_norm)

    t0 = time.time()
    rfs = []
    for i in range(len(METRIC_KEYS)):
        rf = RandomForestRegressor(n_estimators=RF_N_TREES, n_jobs=-1,
                                    random_state=0, max_depth=RF_MAX_DEPTH)
        rf.fit(Xfit, Yfit[:, i])
        rfs.append(rf)
    print(f"  [rf] fit {len(rfs)} single-output RFs in {time.time()-t0:.1f}s")

    Acan_flat = candidates_action.reshape(candidates_action.shape[0], -1)
    Xcan = sx.transform(Acan_flat)
    pred_scaled = np.stack([rf.predict(Xcan) for rf in rfs], axis=1)
    pred_norm   = sy.inverse_transform(pred_scaled)
    raw_pred    = np.zeros_like(pred_norm)
    for i, k in enumerate(METRIC_KEYS):
        lo, hi = M_MIN[i], M_MAX[i]
        raw_pred[:, i] = ((pred_norm[:, i] + 1.0) * 0.5) * (hi - lo) + lo

    def axis_values(name, arr):
        if name == "cpu_mean":   return arr[:, :20].mean(axis=1)
        if name == "K_active":   return (arr[:, :20] > 50).sum(axis=1).astype(np.float32)
        if name in METRIC_KEYS:  return arr[:, METRIC_KEYS.index(name)]
        raise ValueError(name)

    rx = axis_values(x_metric, ref_metric_raw)
    ry = axis_values(y_metric, ref_metric_raw)
    cx = axis_values(x_metric, raw_pred)
    cy = axis_values(y_metric, raw_pred)
    pts = np.stack([rx, ry], axis=1)

    delaunay = None
    try:
        ConvexHull(pts);  delaunay = Delaunay(pts)
    except Exception: pass

    x_lo, x_hi = float(np.percentile(rx, 1)), float(np.percentile(rx, 99))
    y_lo, y_hi = float(np.percentile(ry, 1)), float(np.percentile(ry, 99))
    gx = np.linspace(x_lo, x_hi, grid_n + 1)
    gy = np.linspace(y_lo, y_hi, grid_n + 1)
    cell_count, _, _ = np.histogram2d(rx, ry, bins=[gx, gy])
    cell_count = cell_count.T

    centers_x = (gx[:-1] + gx[1:]) / 2;  centers_y = (gy[:-1] + gy[1:]) / 2
    CX, CY = np.meshgrid(centers_x, centers_y)
    centers = np.stack([CX.ravel(), CY.ravel()], axis=1)
    in_hull = (delaunay.find_simplex(centers) >= 0).reshape(grid_n, grid_n) \
              if delaunay is not None else np.ones_like(cell_count, dtype=bool)
    in_hull_counts = cell_count[in_hull]
    thresh = float(np.percentile(in_hull_counts, empty_pct)) if len(in_hull_counts) else 0.0
    empty_mask = in_hull & (cell_count <= thresh)
    n_empty = int(empty_mask.sum())

    cand_col = np.clip(np.searchsorted(gx, cx) - 1, 0, grid_n - 1)
    cand_row = np.clip(np.searchsorted(gy, cy) - 1, 0, grid_n - 1)

    rng = np.random.default_rng(0)
    cell_to_cand = {}
    for i in range(len(cx)):
        r, c = int(cand_row[i]), int(cand_col[i])
        if empty_mask[r, c]:
            cell_to_cand.setdefault((r, c), []).append(i)
    selected, covered_cells = [], 0
    for (r, c), idxs in cell_to_cand.items():
        if len(selected) >= K: break
        pick = int(idxs[rng.integers(0, len(idxs))])
        selected.append(pick); covered_cells += 1
    print(f"  [hullfill] axes=({x_metric},{y_metric})  empty_cells={n_empty}  "
          f"covered_by_candidate={covered_cells}  selected={len(selected)}/{K}")

    if len(selected) < K:
        used = set(selected)
        leftover_idx = [i for i in range(len(cx)) if i not in used]
        if leftover_idx:
            leftover_pred = np.stack([cx[leftover_idx], cy[leftover_idx]], axis=1)
            anchor = np.vstack([pts, np.stack([cx[selected], cy[selected]], axis=1)]) if selected else pts
            chunk = 1024
            dist_anchor = np.full(len(leftover_pred), np.inf)
            for s in range(0, len(leftover_pred), chunk):
                e = min(s + chunk, len(leftover_pred))
                d = np.linalg.norm(leftover_pred[s:e, None, :] - anchor[None, :, :], axis=-1)
                dist_anchor[s:e] = d.min(axis=1)
            order = np.argsort(-dist_anchor)
            for j in order:
                if len(selected) >= K: break
                selected.append(leftover_idx[int(j)])
            print(f"  [hullfill] filled remaining with {len(selected) - covered_cells} FPS-novel from leftover")
    return np.array(selected)


# ───── 28 single-output RFs + FPS scoring (legacy fallback) ─────
def select_top_K(candidates_action, ref_action, ref_metric_raw, K):
    """Returns indices into candidates_action of the top-K most novel."""
    # Normalize metric to [-1, 1] via RANGES so the RF target matches the
    # clean_trace the diffusion model uses.
    ref_metric_norm = ((ref_metric_raw - M_MIN) / M_RNG) * 2.0 - 1.0   # (N_ref, 28)
    Aref_flat = ref_action.reshape(ref_action.shape[0], -1)             # (N_ref, 380)

    sx = StandardScaler().fit(Aref_flat)
    sy = StandardScaler().fit(ref_metric_norm)
    Xfit = sx.transform(Aref_flat); Yfit = sy.transform(ref_metric_norm)

    t0 = time.time()
    # Fit with oob_score=True → free per-RF held-out R² estimate.
    rfs, r2s = [], []
    for i in range(len(METRIC_KEYS)):
        rf = RandomForestRegressor(n_estimators=RF_N_TREES, n_jobs=-1,
                                    random_state=0, max_depth=RF_MAX_DEPTH,
                                    oob_score=True, bootstrap=True)
        rf.fit(Xfit, Yfit[:, i])
        rfs.append(rf)
        r2s.append(float(rf.oob_score_))
    print(f"  [rf] fit {len(rfs)} single-output RFs in {time.time()-t0:.1f}s")

    # ---- (1) Drop low-R² metrics from novelty / FPS computation ----
    R2_THRESH = float(os.environ.get("R2_THRESH", "0.7"))
    good_dims = [i for i, r in enumerate(r2s) if r >= R2_THRESH]
    bad_dims  = [(METRIC_KEYS[i], r2s[i]) for i in range(len(METRIC_KEYS)) if i not in good_dims]
    print(f"  [r2 mask] kept {len(good_dims)}/{len(METRIC_KEYS)} dims (R² ≥ {R2_THRESH}); "
          f"dropped: {[(k, round(r,2)) for k, r in bad_dims[:6]]}{' …' if len(bad_dims)>6 else ''}")

    # Predict full 28-D, then slice to good_dims for scoring
    Acan_flat = candidates_action.reshape(candidates_action.shape[0], -1)
    Xcan = sx.transform(Acan_flat)
    pred_norm_scaled = np.stack([rf.predict(Xcan) for rf in rfs], axis=1)  # (N_cand, 28)
    pred_norm        = sy.inverse_transform(pred_norm_scaled)              # [-1, 1]
    pred_01_full     = np.clip((pred_norm + 1.0) * 0.5, 0.0, 1.0)            # [0, 1]

    ref_norm_01_full = (ref_metric_norm + 1.0) * 0.5

    pred_01  = pred_01_full[:, good_dims]
    ref_01   = ref_norm_01_full[:, good_dims]

    # ---- (2) Standardize each used dim by IQR of the existing pool so 1 unit ≈ 1 unit ----
    q25 = np.percentile(ref_01, 25, axis=0)
    q75 = np.percentile(ref_01, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-3)
    pred_S = pred_01 / iqr
    ref_S  = ref_01  / iqr

    # KDE rarity in standardized good-dim space
    try:
        kde = gaussian_kde(ref_S.T)
        log_rarity = -np.log(kde.evaluate(pred_S.T) + 1e-12)
    except Exception:
        log_rarity = np.zeros(len(pred_S))

    # R²-weighted tree-variance uncertainty: only on good dims, weighted by their R²
    sv = sy.scale_
    uncert = np.zeros(len(pred_S))
    for i in good_dims:
        tp = np.stack([t.predict(Xcan) for t in rfs[i].estimators_], axis=1)
        uncert += r2s[i] * np.log1p(tp.var(axis=1) * (sv[i] ** 2))

    novelty = log_rarity + LAM_UNCERT * uncert
    nov_n   = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-12)

    # FPS in standardized good-dim space, weighted by (1 + novelty_norm)
    chunk = 512
    dist_ref = np.full(len(pred_S), np.inf)
    for s in range(0, len(pred_S), chunk):
        e = min(s + chunk, len(pred_S))
        d = np.linalg.norm(pred_S[s:e, None, :] - ref_S[None, :, :], axis=-1)
        dist_ref[s:e] = d.min(axis=1)
    weighted = dist_ref * (1.0 + nov_n)

    first = int(np.argmax(weighted))
    sel = [first]
    d_to_sel = np.linalg.norm(pred_S - pred_S[first], axis=1)
    combined = np.minimum(d_to_sel, weighted)
    for _ in range(K - 1):
        combined[sel] = -np.inf
        nxt = int(np.argmax(combined))
        sel.append(nxt)
        d_new = np.linalg.norm(pred_S - pred_S[nxt], axis=1)
        d_to_sel = np.minimum(d_to_sel, d_new)
        combined = np.minimum(d_to_sel, weighted)
    return np.array(sel)


# ───── prev-action variant sampling ─────
def _fps_indices(X, k):
    """Farthest-point sampling: k row indices of X spread across the normalized
    metric space (maximin → reaches the extremes)."""
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


def _prev_filter_axes(ref_M):
    """Return (llc, bw, io) 1-D arrays per pool sample from ref_M (N, 28)."""
    idx = {k: i for i, k in enumerate(METRIC_KEYS)}
    def _col(name): return ref_M[:, idx[name]] if name in idx else np.zeros(ref_M.shape[0])
    def _sum_cols(names): return sum(_col(n) for n in names)
    llc = _col("l3_cache_usage")
    bw  = _sum_cols(("memory_bandwidth_read", "memory_bandwidth_write"))
    io  = _sum_cols(("io_read", "io_write"))
    return llc, bw, io


def build_prev_variants(curr_TS_list, ref_A, ref_M,
                        k=None, shortlist_mult=None,
                        high_resource_frac=None,
                        seed=0):
    """For each chosen curr, build one variant plan [prev, curr] per drawn prev.

    Pipeline:
      1. high-resource filter: union of top `high_resource_frac` by LLC, BW, and IO.
      2. FPS-diverse shortlist over the (LLC, BW, IO) subspace within that subset
         (~k * shortlist_mult prevs).
      3. draw k prevs at random per curr → k variant plans per curr.

    Args:
      curr_TS_list: list of (T=20, S=20) list-of-lists curr-action windows
                     (already write-ready).
      ref_A:  (N_pool, S, T) reference pool of collected NONE-CURRENT actions.
      ref_M:  (N_pool, len(METRIC_KEYS)) matching metric vectors.

    Returns list of [prev_TS, curr_TS] 2-window plans, both (T, S) lists, ready
    for write_actions_to_execution_plans. Returns [] if k<=0 or pool empty.
    """
    k              = K_PREV_VARIANTS     if k              is None else k
    shortlist_mult = PREV_SHORTLIST_MULT if shortlist_mult is None else shortlist_mult
    high_resource_frac  = PREV_HIGH_RESOURCE_FRAC  if high_resource_frac  is None else high_resource_frac

    if k <= 0 or ref_A is None or ref_A.shape[0] == 0:
        return []

    N_pool = ref_A.shape[0]
    llc, bw, io = _prev_filter_axes(ref_M)

    # 1. high-resource union across LLC, BW, IO
    keep = max(k, int(np.ceil(N_pool * high_resource_frac)))
    top_llc = set(np.argsort(llc)[-keep:].tolist())
    top_bw  = set(np.argsort(bw)[-keep:].tolist())
    top_io  = set(np.argsort(io)[-keep:].tolist())
    hi_idx  = np.array(sorted(top_llc | top_bw | top_io), dtype=int)

    # 2. FPS shortlist over (LLC, BW, IO) inside the subset
    axis_sub = np.stack([llc[hi_idx], bw[hi_idx], io[hi_idx]], axis=1)
    n_short = min(len(hi_idx), max(k, k * shortlist_mult))
    short_local = _fps_indices(axis_sub, n_short)
    short_idxs = [int(hi_idx[i]) for i in short_local]

    llcs = [float(llc[i]) for i in short_idxs]
    bws  = [float(bw[i])  for i in short_idxs]
    ios  = [float(io[i])  for i in short_idxs]
    print(f"  [prev] pool={N_pool}, hi-subset={len(hi_idx)}, "
          f"shortlist={n_short} (k={k}, mult={shortlist_mult})")
    print(f"  [prev] shortlist LLC {min(llcs):.0f}-{max(llcs):.0f}  "
          f"BW {min(bws):.1f}-{max(bws):.1f}  IO {min(ios):.0f}-{max(ios):.0f}")

    # 3. per-curr random draw of k prevs
    rng = random.Random(seed)
    variants = []
    for curr_TS in curr_TS_list:
        for prev_idx in rng.sample(short_idxs, min(k, len(short_idxs))):
            prev_TS = ref_A[prev_idx].T.tolist()   # (S, T) → (T, S) write layout
            variants.append([prev_TS, curr_TS])
    print(f"  [prev] emitted {len(variants)} variant plans "
          f"({len(curr_TS_list)} currs × {k} prevs)")
    return variants


# ───── dispatch helpers ─────
def write_round(round_idx, plans_list):
    """plans_list: iterable of [prev_window, curr_window] python lists where each
    window is (T=20, S=20) list-of-lists. A prev_window equal to NONE_PREV
    (all-zero) is dropped by write_actions_to_execution_plans so none-current
    plans become 1-window h5; real prev-variant plans stay 2-window."""
    dest = f"{OUT_DIR}/round_{round_idx}"
    os.makedirs(dest, exist_ok=True)
    write_actions_to_execution_plans(list(plans_list), dest, HOSTS)
    n_h5 = len(glob.glob(f"{dest}/chunk_*/plans/plan_*.h5"))
    print(f"  [write] round_{round_idx}: {n_h5} h5 files across {H} chunks")
    return dest


def _wrap_none_current(actions_STK):
    """Wrap (K, S, T) actions as list of [NONE_PREV, (T, S) list-of-lists]."""
    return [[NONE_PREV, sa.T.tolist()] for sa in actions_STK]


def preclean_workers():
    """Delete stale validation-*.zip in the remote home dir on every worker;
    the parallel puller must not grab a previous round's zips."""
    for h in HOSTS:
        subprocess.run(
            ["ssh", "-i", worker_config.PRIVATE_KEY_PATH,
             "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
             f"{worker_config.USERNAME}@{h}",
             f"rm -f {worker_config.REMOTE_HOME_DIR}/validation-*.zip"],
            capture_output=True, text=True, timeout=15)


def manual_pull_zips(round_dir):
    """One-shot SCP of validation-N.zip from each worker into round_dir/chunk_N/."""
    for c, h in enumerate(HOSTS):
        dst = f"{round_dir}/chunk_{c}/validation-{c}.zip"
        if os.path.exists(dst): continue
        subprocess.run(
            ["scp", "-i", worker_config.PRIVATE_KEY_PATH,
             "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             f"{worker_config.USERNAME}@{h}:{worker_config.REMOTE_HOME_DIR}/validation-{c}.zip", dst],
            capture_output=True, text=True, timeout=30)
        if os.path.exists(dst):
            subprocess.run(["unzip", "-oq", dst], cwd=os.path.dirname(dst))
    n_stats = len(glob.glob(f"{round_dir}/chunk_*/results/stats-plan_*.txt"))
    print(f"  [pull] {n_stats} stats files in {round_dir}")
    return n_stats


import threading
def start_parallel_puller(round_dir, stop_event):
    """Background thread: every 30 s, SCP any newly-ready validation-N.zip into
    round_dir/chunk_N/, unblocking profile_actions's polling loop on workers
    that can't resolve the controller hostname to push results back."""
    def _loop():
        while not stop_event.is_set():
            for c, h in enumerate(HOSTS):
                dst = f"{round_dir}/chunk_{c}/validation-{c}.zip"
                if os.path.exists(dst): continue
                r = subprocess.run(
                    ["scp", "-i", worker_config.PRIVATE_KEY_PATH,
                     "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                     f"{worker_config.USERNAME}@{h}:{worker_config.REMOTE_HOME_DIR}/validation-{c}.zip", dst],
                    capture_output=True, text=True, timeout=30)
                if r.returncode == 0 and os.path.exists(dst):
                    subprocess.run(["unzip", "-oq", dst], cwd=os.path.dirname(dst))
                    print(f"  [puller] grabbed chunk_{c} from {h.split('.')[0]}")
            if stop_event.wait(timeout=30): break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ───── main loop ─────
def main():
    profiler = Profiler(InitializeRequest(
        user_name=worker_config.USERNAME,
        private_key_path=worker_config.PRIVATE_KEY_PATH,
        worker_host_names=HOSTS,
        my_hostname=worker_config.MY_HOSTNAME,
    ))

    # Round 0: curr-only one-hot sweep, no REF/selector. Prev-variants only
    # fire from round_1 once a pool exists.
    r0_chosen = seed_round0()
    if r0_chosen is not None:
        round_dir = f"{OUT_DIR}/round_0"
        print(f"[round_0] selected K_thread mean=⟨{((r0_chosen.sum(1) > 0.5).sum(1)).mean():.2f}⟩  "
              f"K_stressor mean=⟨{((r0_chosen.sum(2) > 0.5).sum(1)).mean():.2f}⟩  "
              f"max_w=⟨{r0_chosen.max(axis=(1,2)).mean():.3f}⟩")
        write_round(0, _wrap_none_current(r0_chosen))
        print(f"[round_0] pre-cleaning stale validation zips on workers …")
        preclean_workers()
        stop_event = threading.Event()
        puller = start_parallel_puller(round_dir, stop_event)
        print(f"[round_0] running Profiler on {H} hosts (parallel puller live) …")
        try:
            asyncio.run(profiler.profile_actions(round_dir, skip_parsing=True))
        finally:
            stop_event.set()
            puller.join(timeout=60)
        manual_pull_zips(round_dir)
        np.save(f"{round_dir}/chosen_actions.npy", r0_chosen)
        print(f"[round_0] done")

    # Total plans per round = curr-only (N_SELECT) + K prev-variants each.
    plans_per_round = N_SELECT * (1 + K_PREV_VARIANTS)

    for r in range(1, N_ROUNDS_NEW + 1):
        round_dir = f"{OUT_DIR}/round_{r}"
        if os.path.exists(round_dir):
            n_stats = len(glob.glob(f"{round_dir}/chunk_*/results/stats-plan_*.txt"))
            # chosen_actions.npy is written at the very end of a round, so its
            # presence marks the round done regardless of the config it ran
            # under (plan counts differ across configs). The stats count is a
            # fallback for rounds missing the marker.
            done_marker = os.path.exists(f"{round_dir}/chosen_actions.npy")
            if done_marker or n_stats >= plans_per_round * 0.9:
                reason = "marker" if done_marker else f"{n_stats}/{plans_per_round} stats"
                print(f"\n=== round_{r} already done ({reason}) — skipping ===")
                continue

        print(f"\n=== Round {r} ===")
        # 1. Build REF from rounds 0..r-1
        print(f"[round_{r}] loading reference …")
        ref_A, ref_M = load_ref_through_round(r - 1)
        print(f"[round_{r}] REF: {ref_A.shape[0]} (action, metric) pairs")

        # 2. Candidate generation — alternate between FPS-random and union-
        #    superposition by round parity. Odd rounds (1, 3, 5, …) sample
        #    random plans; even rounds (2, 4, 6, …) build thread-disjoint
        #    unions of existing REF pairs. K-stratified selector runs on
        #    whichever pool the round emitted.
        GRID_N = int(os.environ.get("GRID_N", "20"))
        EMPTY_PCT = float(os.environ.get("EMPTY_PCT", "20"))
        mode = "fps" if r % 2 == 1 else "union"
        print(f"[round_{r}] mode={mode}")

        if mode == "fps":
            print(f"[round_{r}] sampling {N_CANDIDATES} candidates via sample_plans")
            candidates = sample_plans(N_CANDIDATES, seed=42 + r)
        else:
            U_PER_CELL = int(os.environ.get("UNION_N_PER_CELL", "3"))
            U_MAX_K    = int(os.environ.get("UNION_MAX_K", str(NUM_THREADS)))
            print(f"[round_{r}] union candidates (n_per_cell={U_PER_CELL}, "
                  f"max_K={U_MAX_K}) per-pair emission:")
            candidates = generate_union_candidates(
                ref_A, ref_M, HULL_PAIRS, GRID_N, EMPTY_PCT,
                n_per_cell=U_PER_CELL, max_K=U_MAX_K, seed=2026 + r,
            )
            print(f"[round_{r}] union total: {len(candidates)}")
            if len(candidates) == 0:
                print(f"[round_{r}] no union candidates emitted — falling back to sample_plans")
                candidates = sample_plans(N_CANDIDATES, seed=42 + r)

        # Cap per-thread sum_w to 1.0 so the binary's scheduler doesn't throttle.
        before = (candidates.sum(axis=1) > 1.0).any(axis=1).sum()
        candidates = _cap_sumw_per_thread(candidates, cap=1.0)
        print(f"[round_{r}] capped sum_w → 1.0 (had {before}/{len(candidates)} "
              f"candidates with any over-budget thread)")

        # 3. Selector: K-stratified hull (default if K_STRATIFY=1), multi-pair
        #    hull, or single-pair (MULTI_PAIRS=0). K_STRATIFY allocates budget
        #    per K_active bucket so the data covers K=1..20 uniformly.
        if os.environ.get("K_STRATIFY", "0") != "0":
            K_MIN = int(os.environ.get("K_STRAT_MIN", "1"))
            K_MAX = int(os.environ.get("K_STRAT_MAX", "20"))
            print(f"[round_{r}] K-stratified hullfill (K={K_MIN}..{K_MAX}, "
                  f"{len(HULL_PAIRS)} pairs)  grid={GRID_N}x{GRID_N}  "
                  f"empty_thresh={EMPTY_PCT}%-ile")
            sel_idx = select_top_K_hullfill_kstrat(
                candidates, ref_A, ref_M, N_SELECT,
                pairs=HULL_PAIRS, grid_n=GRID_N, empty_pct=EMPTY_PCT,
                K_min=K_MIN, K_max=K_MAX,
            )
        elif os.environ.get("MULTI_PAIRS", "1") != "0":
            print(f"[round_{r}] multi-pair hullfill ({len(HULL_PAIRS)} pairs)  "
                  f"grid={GRID_N}x{GRID_N}  empty_thresh={EMPTY_PCT}%-ile")
            sel_idx = select_top_K_hullfill_multi(
                candidates, ref_A, ref_M, N_SELECT,
                pairs=HULL_PAIRS, grid_n=GRID_N, empty_pct=EMPTY_PCT,
            )
        else:
            HULL_X = os.environ.get("HULL_X", "cpu_mean")
            HULL_Y = os.environ.get("HULL_Y", "io_read")
            print(f"[round_{r}] hullfill: ({HULL_X},{HULL_Y})  "
                  f"grid={GRID_N}x{GRID_N}  empty_thresh={EMPTY_PCT}%-ile")
            sel_idx = select_top_K_hullfill(
                candidates, ref_A, ref_M, N_SELECT,
                x_metric=HULL_X, y_metric=HULL_Y,
                grid_n=GRID_N, empty_pct=EMPTY_PCT,
            )
        chosen  = candidates[sel_idx]
        print(f"[round_{r}] selected K_thread mean=⟨{((chosen.sum(1) > 0.5).sum(1)).mean():.2f}⟩  "
              f"K_stressor mean=⟨{((chosen.sum(2) > 0.5).sum(1)).mean():.2f}⟩  "
              f"max_w=⟨{chosen.max(axis=(1,2)).mean():.3f}⟩")

        # 3b. Optional K prev-variants per curr (see build_prev_variants).
        #     Interleave contiguous-per-curr so each curr's plans land on the
        #     SAME worker chunk (write_actions_to_execution_plans slices the
        #     batch sequentially across HOSTS); same-host NONE↔PREV removes
        #     the host-id confound from carry-over measurement.
        curr_TS_list = [sa.T.tolist() for sa in chosen]     # (T=20, S=20)
        none_plans   = [[NONE_PREV, c] for c in curr_TS_list]
        prev_variants = build_prev_variants(curr_TS_list, ref_A, ref_M, seed=r) \
                        if K_PREV_VARIANTS > 0 else []

        batch = []
        for i in range(len(curr_TS_list)):
            batch.append(none_plans[i])
            batch.extend(prev_variants[i * K_PREV_VARIANTS : (i + 1) * K_PREV_VARIANTS])
        if prev_variants:
            print(f"[round_{r}] batch: {len(none_plans)} curr-only + {len(prev_variants)} "
                  f"prev-variants = {len(batch)} plans (contiguous-per-curr, "
                  f"same-host NONE↔PREV)")

        # 4. Write h5 + dispatch + start parallel puller before profile_actions
        #    (otherwise profile_actions blocks waiting for validation-N.zip)
        write_round(r, batch)
        print(f"[round_{r}] pre-cleaning stale validation zips on workers …")
        preclean_workers()
        stop_event = threading.Event()
        puller = start_parallel_puller(round_dir, stop_event)
        print(f"[round_{r}] running Profiler on {H} hosts (parallel puller live) …")
        try:
            asyncio.run(profiler.profile_actions(round_dir, skip_parsing=True))
        finally:
            stop_event.set()
            puller.join(timeout=60)

        # 5. Final sweep (catches any zips that landed between puller iterations)
        manual_pull_zips(round_dir)

        # chosen_actions.npy doubles as the round-done marker
        np.save(f"{round_dir}/chosen_actions.npy", chosen)
        print(f"[round_{r}] done")

    print(f"\nAll {N_ROUNDS_NEW} rounds done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
