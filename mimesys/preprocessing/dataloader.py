import concurrent.futures
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Curr-only ("none-current") plan reconstruction
# ---------------------------------------------------------------------------
# A none-current plan skips profiling its (empty) prev window, so the plan H5
# holds a single (curr) window. Here we reconstruct the canonical 2-window form
# for training: prepend NONE_PREV (an all-zero "no previous action") as the prev
# action and prepend an all-zero prev metric slot so curr stays at slot 1
# (matching fully-profiled 2-window plans). Kept in sync with NONE_PREV in
# mimesys/collection/collect_training_data.py (duplicated to avoid a circular
# import: that module imports parsers from this one).
_NUM_THREADS, _NUM_STRESSORS = 20, 19
_NONE_PREV_T = [[0.0] * _NUM_THREADS for _ in range(_NUM_STRESSORS)]   # [stressor][thread]

def _prepend_zero_metric_slot(vals, cycle):
    """Insert a zero at the front of every `cycle`-length group in the flat
    per-metric list, turning an N-slot capture into an (N+1)-slot one whose new
    leading slot is all-zero (the un-profiled prev window).

    Drops the FIRST `cycle`-block of `vals` (curr_0 + sleep_0). curr_0 contains
    warmup-tail contamination from the kernel-init/populate phase that bleeds
    into the first measured slot. With MIMESYS_ITERS=4 the kernel reports 4
    curr slots and we keep the last 3 (curr_1..curr_3); with ITERS=3 we keep 2.
    Set MIMESYS_KEEP_FIRST_CURR=1 to disable the drop (legacy v11 compat).
    """
    import os
    drop_first = os.environ.get("MIMESYS_KEEP_FIRST_CURR", "0") != "1"
    if drop_first and len(vals) > cycle:
        vals = vals[cycle:]
    out = []
    for i in range(0, len(vals), cycle):
        out.append(0.0)
        out.extend(vals[i:i + cycle])
    return out


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------
# Metric vector layout (sorted alphabetically, 23-dim):
#   [0:10]  avg_cpu_utilizations_core_00..09  (socket 0)
#   [10:20] avg_cpu_utilizations_core_10..19  (socket 1)
#   [20]    io
#   [21]    l3_cache_usage    (socket-aggregated)
#   [22]    memory_bandwidth  (socket-aggregated)
#
# Action label layout (after transpose): [stressors, threads=20]
#   columns [0:10]  = socket 0 threads
#   columns [10:20] = socket 1 threads

def _apply_perm_to_trace(trace: np.ndarray, perm_s0, perm_s1) -> np.ndarray:
    out = trace.copy()
    out[0:10]  = trace[perm_s0]
    out[10:20] = trace[[10 + p for p in perm_s1]]
    return out


def _apply_swap_to_trace(trace: np.ndarray) -> np.ndarray:
    out = trace.copy()
    out[0:10]  = trace[10:20]
    out[10:20] = trace[0:10]
    # LLC/BW are socket-aggregated → no per-socket swap needed.
    return out


def _apply_perm_to_label(label, perm_s0, perm_s1):
    full_perm = perm_s0 + [10 + p for p in perm_s1]
    return [[row[p] for p in full_perm] for row in label]


def _apply_swap_to_label(label):
    return [row[10:20] + row[0:10] for row in label]


def augment_dataset(data: list, aug_factor: int, intra_only: bool = False,
                    high_io_aug_factor: int = 1, io_raw_threshold: float = 1000.0) -> list:
    """
    Expand dataset by aug_factor using intra-socket permutation and (optionally) socket swap.

    If intra_only=False (default), for each original sample (aug_factor - 1) variants are added:
      - variants 0..half-1         : intra-socket perm only
      - variant  half              : socket swap only
      - variants half+1..end       : intra-socket perm + socket swap

    If intra_only=True, all (aug_factor - 1) variants use intra-socket permutation only.
    No socket swap is applied. Physically valid because threads on the same socket share
    the same DRAM controller — socket-level totals (LLC, BW) are unchanged by permutation.

    If high_io_aug_factor > 1, samples whose raw IO metric (collated_trace[20]) exceeds
    io_raw_threshold KB/s receive high_io_aug_factor - 1 additional intra-socket-only
    variants, oversampling IO-heavy examples to address class imbalance.

    Both action (label / prev_label) and metric trace are augmented consistently.
    """
    if aug_factor <= 1 and high_io_aug_factor <= 1:
        return data

    augmented = list(data)
    n_extra = aug_factor - 1
    half = n_extra // 2

    for item in data:
        trace     = np.array(item["clean_trace"])
        raw_trace = np.array(item["info"]["collated_trace"])
        label     = item["label"]           # list[stressor][thread]
        prev_lbl  = item["prev_label"]      # list[stressor][thread] or None
        coll_lbl  = item["info"]["collated_label"]

        for aug_idx in range(n_extra):
            do_perm = True if intra_only else (aug_idx != half)
            do_swap = False if intra_only else (aug_idx >= half)

            if do_perm:
                perm_s0 = np.random.permutation(10).tolist()
                perm_s1 = np.random.permutation(10).tolist()
                new_trace = _apply_perm_to_trace(trace, perm_s0, perm_s1)
                new_raw   = _apply_perm_to_trace(raw_trace, perm_s0, perm_s1)
                new_lbl   = _apply_perm_to_label(label, perm_s0, perm_s1)
                new_prev  = _apply_perm_to_label(prev_lbl, perm_s0, perm_s1) if prev_lbl is not None else None
                new_coll  = _apply_perm_to_label(coll_lbl, perm_s0, perm_s1)
            else:
                new_trace = trace.copy()
                new_raw   = raw_trace.copy()
                new_lbl   = [list(row) for row in label]
                new_prev  = ([list(row) for row in prev_lbl] if prev_lbl is not None else None)
                new_coll  = [list(row) for row in coll_lbl]

            if do_swap:
                new_trace = _apply_swap_to_trace(new_trace)
                new_raw   = _apply_swap_to_trace(new_raw)
                new_lbl   = _apply_swap_to_label(new_lbl)
                if new_prev is not None:
                    new_prev = _apply_swap_to_label(new_prev)
                new_coll  = _apply_swap_to_label(new_coll)

            augmented.append({
                "clean_trace": new_trace.tolist(),
                "label":       new_lbl,
                "prev_label":  new_prev,
                "info": {
                    "collated_trace": new_raw.tolist(),
                    "collated_label": new_coll,
                },
            })

    # --- high-IO oversampling (intra-socket only) ---
    IO_IDX = 20
    if high_io_aug_factor > 1:
        n_hi_extra = high_io_aug_factor - 1
        hi_count = 0
        for item in data:
            raw_trace = np.array(item["info"]["collated_trace"])
            if raw_trace[IO_IDX] < io_raw_threshold:
                continue
            hi_count += 1
            trace    = np.array(item["clean_trace"])
            label    = item["label"]
            prev_lbl = item["prev_label"]
            coll_lbl = item["info"]["collated_label"]
            for _ in range(n_hi_extra):
                perm_s0   = np.random.permutation(10).tolist()
                perm_s1   = np.random.permutation(10).tolist()
                new_trace = _apply_perm_to_trace(trace, perm_s0, perm_s1)
                new_raw   = _apply_perm_to_trace(raw_trace, perm_s0, perm_s1)
                new_lbl   = _apply_perm_to_label(label, perm_s0, perm_s1)
                new_prev  = (_apply_perm_to_label(prev_lbl, perm_s0, perm_s1)
                             if prev_lbl is not None else None)
                new_coll  = _apply_perm_to_label(coll_lbl, perm_s0, perm_s1)
                augmented.append({
                    "clean_trace": new_trace.tolist(),
                    "label":       new_lbl,
                    "prev_label":  new_prev,
                    "info": {
                        "collated_trace": new_raw.tolist(),
                        "collated_label": new_coll,
                    },
                })
        print(f"High-IO aug (>{io_raw_threshold:.0f} KB/s): {hi_count} originals "
              f"× {n_hi_extra} extra = {hi_count * n_hi_extra} new samples added")

    return augmented

from mimesys.schema.stressor_action import FleetBenchAction
from mimesys.preprocessing.system_trace import get_min_max, normalize_trace
from mimesys.preprocessing.pqos_parser import pqos_metrics_dict, PQOS_METRIC_KEYS
from mimesys.preprocessing.parsers import (
    get_tacc_stats_and_energy_from_benchmark_name,
    parse_trace_file,
    process_trace,
    process_trace_fine_grained,
    process_trace_granular,
    process_trace_all,
)


# ---------------------------------------------------------------------------
# Pure data utilities
# ---------------------------------------------------------------------------

def _metrics_matrix(metrics_by_name: dict[str, list[list[float]]], sample_idx: int) -> np.ndarray:
    """Return (num_metrics, time_steps) array for one sample, metrics sorted by name."""
    return np.array([metrics_by_name[k][sample_idx] for k in sorted(metrics_by_name)])


# ---------------------------------------------------------------------------
# File-level readers
# ---------------------------------------------------------------------------

def read_metric_file_from_real_app(file_path: str) -> list[dict]:
    """Parse a TACC stats file and return per-window metric dicts."""
    _, parsed_traces = parse_trace_file(file_path)

    period = 5
    window_size = 30
    step_size = 30

    trace_windows = []
    for start_idx in range(0, len(parsed_traces), step_size):
        end_idx = start_idx + window_size
        if end_idx <= len(parsed_traces):
            window = parsed_traces[start_idx:end_idx]
        else:
            window = parsed_traces[start_idx:]
            window.extend([defaultdict(int)] * (end_idx - len(parsed_traces)))
        trace_windows.append(window)

    # Real-app pqos collected as a paired pqos-N_X-mix_Y.log file alongside
    # stats-N_X-mix_Y.txt. Load the full per-second pqos series once for the
    # whole file, then slice it into the same 30-sec windows used for stats.
    pqos_path = file_path.replace("/stats", "/pqos").replace("stats-", "pqos-").replace(".txt", ".log")
    try:
        full_pq = pqos_metrics_dict(pqos_path) if Path(pqos_path).exists() else {}
    except Exception:
        full_pq = {}

    metrics_list = []
    for win_idx, window in enumerate(trace_windows):
        output = process_trace_fine_grained(window, period=2, duration=29)
        if output is not None:
            # Drop redundant combined keys to mirror training-side process_row.
            output.pop("io", None)
            output.pop("memory_bandwidth", None)
            n_steps = len(next(iter(output.values()))) if output else 0
            # Aggregate per-second pqos over each 30-sec window into period-2 buckets so
            # length matches the stats-side n_steps. Falls back to zero-fill if missing.
            win_start = win_idx * step_size
            win_end = min(win_start + window_size, len(full_pq.get(PQOS_METRIC_KEYS[0], [])))
            for k in PQOS_METRIC_KEYS:
                series = full_pq.get(k, [])
                if win_end > win_start and series:
                    chunk = series[win_start:win_end]
                    # bucket-mean over period=2 to match process_trace_fine_grained's stride
                    if len(chunk) >= 2 * n_steps:
                        bucketed = [
                            float(np.mean(chunk[i*2:(i+1)*2])) for i in range(n_steps)
                        ]
                    else:
                        # short tail — pad with zeros to n_steps
                        bucketed = (chunk + [0.0] * n_steps)[:n_steps]
                    output[k] = bucketed
                else:
                    output[k] = [0.0] * n_steps
            metrics_list.append(output)
    return metrics_list


def read_metric_file_per_second(file_path: str) -> list[dict]:
    """Parse a TACC stats file and return ONE per-second metric dict for the whole file.

    Unlike `read_metric_file_from_real_app` (which chops into 30-sec windows and runs
    per-2-sec aggregation inside each), this produces one dict whose lists are length N
    (one entry per second across the full file). This is the granularity the K=3000
    K-means RL sampling expects: each output dict becomes N per-second data points after
    `get_val_datasets_from_metric_data`.
    """
    _, parsed_traces = parse_trace_file(file_path)
    N = len(parsed_traces)
    if N <= 0:
        return []
    metrics_output = process_trace_granular(parsed_traces, period=1, duration=N)
    if metrics_output is None:
        return []
    # Mirror training-side processing: drop redundant combined keys + pair pqos.
    metrics_output.pop("io", None)
    metrics_output.pop("memory_bandwidth", None)
    metrics_output.pop("avg_cpu_utilizations_total", None)
    pqos_path = file_path.replace("/stats", "/pqos").replace("stats-", "pqos-").replace(".txt", ".log")
    try:
        pq = pqos_metrics_dict(pqos_path) if Path(pqos_path).exists() else {}
    except Exception:
        pq = {}
    n_steps = len(next(iter(metrics_output.values()))) if metrics_output else 0
    for k in PQOS_METRIC_KEYS:
        v = pq.get(k, [])
        if len(v) == n_steps:
            metrics_output[k] = list(v)
        else:
            metrics_output[k] = [0.0] * n_steps
    return [metrics_output]


def read_metric_action_datasets_fleetbench(
    file_path: str, round_idx: int, chunk_idx: int, samples_per_chunk: int,
) -> list[dict]:
    chunk_action_path = f"{file_path}/round_{round_idx}/chunk_{chunk_idx}/plans"
    chunk_metric_path = f"{file_path}/round_{round_idx}/chunk_{chunk_idx}/results"

    action_metric_files = [
        (f"{chunk_action_path}/plan_{i:06d}.h5", f"{chunk_metric_path}/stats-plan_{i:06d}.txt")
        for i in range(samples_per_chunk)
        if Path(f"{chunk_action_path}/plan_{i:06d}.h5").exists()
        and Path(f"{chunk_metric_path}/stats-plan_{i:06d}.txt").exists()
    ]

    def process_row(row):
        action_file, metric_file = row
        _, parsed_traces = parse_trace_file(metric_file)
        # metrics_output = process_trace_fine_grained(parsed_traces, period=2, duration=20)
        metrics_output = process_trace_all(parsed_traces)
        if metrics_output is None:
            return None
        # Drop redundant combined keys — r/w split versions carry the same info
        # without double-counting at training time.
        metrics_output.pop("io", None)
        metrics_output.pop("memory_bandwidth", None)
        # Merge pqos (LLC occupancy KB, IPC, L3 misses/sec). libpqos in-process
        # polling produces 1 sample per slot, sample-aligned with hpc, so the
        # arrays slot into the same time-series structure as the hpc keys.
        # Missing pqos.log → empty arrays, which get padded to zeros downstream.
        pqos_file = metric_file.replace("/stats-plan_", "/pqos-plan_").replace(".txt", ".log")
        try:
            pqos_d = pqos_metrics_dict(pqos_file) if Path(pqos_file).exists() else {}
        except Exception:
            pqos_d = {}
        hpc_len = len(next(iter(metrics_output.values())))
        for k in PQOS_METRIC_KEYS:
            v = pqos_d.get(k, [])
            metrics_output[k] = list(v) if len(v) == hpc_len else [0.0] * hpc_len
        action = FleetBenchAction.from_action_file(action_file)
        weights = action.to_2d_list(transpose=True)   # [window][stressor][thread]
        if len(action.weights) == 1:
            # Curr-only (none-current) plan: prev profiling was skipped. Reconstruct
            # [NONE_PREV, curr] and prepend a zero prev metric slot (capture cycled
            # through curr/noop = 2 slots) so curr stays at slot 1.
            metrics_output = {k: _prepend_zero_metric_slot(v, cycle=2)
                              for k, v in metrics_output.items()}
            weights = [_NONE_PREV_T] + weights
        elif len(action.weights) != 2:
            return None
        return dict(metrics_output), weights

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_row, action_metric_files),
            total=len(action_metric_files),
        ))

    valid = [r for r in results if r is not None]
    print(f"Processed {len(valid)} valid samples out of {len(action_metric_files)} files")
    return valid


# ---------------------------------------------------------------------------
# Dataset construction from (metrics, actions) pairs
# ---------------------------------------------------------------------------

def get_val_datasets_from_metric_data(
    metric_datasets: list[dict], metrics_range_dict: dict
) -> list[dict]:
    target_metrics: dict = defaultdict(list)
    for metrics in metric_datasets:
        if metrics is not None:
            for name, vals in metrics.items():
                target_metrics[name].append(vals)

    normalized: dict = defaultdict(list)
    for name, vals_list in target_metrics.items():
        mn, mx = metrics_range_dict[name]
        normalized[name] = normalize_trace(vals_list, mn, mx)

    datasets = []
    for i in range(len(metric_datasets)):
        raw_mat  = _metrics_matrix(target_metrics, i)
        norm_mat = _metrics_matrix(normalized, i)
        T = raw_mat.shape[1]
        for t in range(T):
            if not raw_mat[:, t].any():
                continue
            prev = norm_mat[:, t - 1].tolist() if t > 0 else [0.0] * norm_mat.shape[0]
            datasets.append({
                "clean_trace":         norm_mat[:, t].tolist(),
                "collated_trace":      raw_mat[:, t].tolist(),
                "prev_clean_trace":    prev,
                "prev_collated_trace": (raw_mat[:, t - 1].tolist() if t > 0 else [0.0] * raw_mat.shape[0]),
            })
    return datasets


def get_datasets_from_metric_action_pairs(
    metric_action_datasets: list, max_time_steps: int, include_invalid: bool = False
) -> tuple[list[dict], dict]:
    target_metrics: dict = defaultdict(list)
    labels_list = []
    invalid_indices = []

    for idx, entry in enumerate(metric_action_datasets):
        if entry is not None:
            metrics_output, actions_output = entry
            for name, vals in metrics_output.items():
                target_metrics[name].append(vals)
            labels_list.append(actions_output)
        else:
            invalid_indices.append(idx)

    normalized: dict = defaultdict(list)
    metrics_range_dict = {}
    for name, vals_list in target_metrics.items():
        mn, mx = get_min_max(target_metrics, name)
        normalized[name] = normalize_trace(vals_list, mn, mx)
        metrics_range_dict[name] = (mn, mx)

    def _scale_label(labels):
        return [[(e * 2) - 1 for e in row] for row in labels]

    datasets = []
    n_valid = len(metric_action_datasets) - len(invalid_indices)
    for i in range(n_valid):
        norm_mat  = _metrics_matrix(normalized, i)
        raw_mat   = _metrics_matrix(target_metrics, i)
        for t in range(len(labels_list[i])):
            # Skip the NONE_PREV padding slot inserted by process_row for 1-window
            # plans. Filter is by *identity* (this is literally the module-level
            # _NONE_PREV_T singleton placed at slot 0), so we don't false-positive
            # on real plans whose curr happens to be all-idle.
            if labels_list[i][t] is _NONE_PREV_T:
                continue
            datasets.append({
                "clean_trace": norm_mat[:, t].tolist(),
                "label":       _scale_label(labels_list[i][t]),
                "prev_label":  (_scale_label(labels_list[i][t - 1])
                                if t > 0 and labels_list[i][t - 1] is not _NONE_PREV_T
                                else None),
                "info": {
                    "collated_trace": raw_mat[:, t].tolist(),
                    "collated_label": labels_list[i][t],
                },
            })

    if include_invalid:
        for idx in invalid_indices:
            datasets.insert(idx, None)

    return datasets, metrics_range_dict


def filter_high_variance_data(M, var, threshold=0.1):
    if not len(M):
        return []
    flattened = [sample for samples in M for sample in samples]
    mins = np.min(flattened, axis=0)
    maxs = np.max(flattened, axis=0)
    dynamic_thresholds = (maxs - mins) * threshold
    print("Metrics Range:", maxs - mins)

    # Only check curr_action slot (index 1), consistent with collection-time filter
    # which uses v[num_metrics*1 : num_metrics*2]. Checking the no-op/sleep slot
    # inflates the filter rate because idle-state metrics are inherently noisier.
    bad = [i for i, v in enumerate(var) if len(v) > 1 and np.any(v[1] > dynamic_thresholds)]
    print(
        f"Filtered {len(bad)} samples with high variance "
        f"({len(bad) / len(M) * 100:.2f}%)\n"
        f"Dynamic thresholds: {dynamic_thresholds}"
    )
    return bad


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class StressNgDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.tensor(d["label"], dtype=torch.float32)
        return {
            **d,
            "label": label,
            "clean_trace": torch.tensor(d["clean_trace"], dtype=torch.float32),
            "prev_label": (
                torch.tensor(d["prev_label"], dtype=torch.float32)
                if d["prev_label"] is not None
                else -torch.ones_like(label)
            ),
        }

    def shuffle_data(self):
        random.shuffle(self.data)


class StressNgTestDataset(Dataset):
    def __init__(self, data, label_shape, dataset_size: int = 1024):
        super().__init__()
        self.initial_data = data
        self.label_shape = label_shape
        self.shuffle_data(dataset_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        has_prev = d.get("prev_label") is not None
        return {
            **d,
            "training_data": has_prev,
            "label": torch.randn(self.label_shape, dtype=torch.float32),
            "clean_trace": torch.tensor(d["clean_trace"], dtype=torch.float32),
            "prev_clean_trace": (
                torch.tensor(d["prev_clean_trace"], dtype=torch.float32)
                if d.get("prev_clean_trace") is not None
                else -torch.ones(len(d["clean_trace"]), dtype=torch.float32)
            ),
            "prev_label": (
                torch.tensor(d["prev_label"], dtype=torch.float32)
                if has_prev
                else -torch.ones(self.label_shape, dtype=torch.float32)
            ),
        }

    def shuffle_data(self, dataset_size: int):
        random.shuffle(self.initial_data)
        self.data = self.initial_data[:dataset_size]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class CustomDataLoader(pl.LightningDataModule):
    _NUM_CHUNKS = 14
    _NUM_ROUNDS = 70
    _SAMPLES_PER_CHUNK = 256  # round_0 has 190 plans/chunk (after 19-stressor refresh); Config C variants 112/chunk

    def __init__(
        self,
        file_path: str,
        test_data_path: str,
        max_time_steps: int,
        batch_size: int = 32,
        test_batch_size: int = 4,
        use_rl: bool = False,
        ddpo_batch_size: int = 48,
        aug_factor: int = 1,
        intra_only_aug: bool = False,
        high_io_aug_factor: int = 1,
        io_raw_threshold: float = 1000.0,
        cpu_avg_threshold: float = 50.0,
        llc_max_threshold: float = 100.0,
        bw_max_threshold: float = 5.0,
        rl_train_ratio: float = 0.333,
        rl_high_io_fraction: float = 0.20,
        rl_high_cpu_fraction: float = 0.20,
        rl_high_llc_fraction: float = 0.20,
        rl_high_bw_fraction: float = 0.20,
        rl_sample_method: str = "kmean",   # "kmean" | "grid"
        rl_kmean_k: int = 3000,
        rl_use_high_resource_supplements: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        data, metrics_range_dict = self._load_or_build_training_data(file_path, max_time_steps)
        self.trace_range = metrics_range_dict

        if aug_factor > 1 or high_io_aug_factor > 1:
            mode = "intra-socket only" if intra_only_aug else "intra-socket + socket-swap"
            print(f"Augmenting training data by {aug_factor}x ({mode}), "
                  f"high-IO by {high_io_aug_factor}x ...")
            data = augment_dataset(data, aug_factor, intra_only=intra_only_aug,
                                   high_io_aug_factor=high_io_aug_factor,
                                   io_raw_threshold=io_raw_threshold)
            print(f"Augmented dataset size: {len(data)}")

        # ── Tier 1.1: density filter on training samples ──────────────────────
        # MIMESYS_DENSITY_MIN_ACTIVE_CORES (default 0, no filter): drop samples
        # where fewer than this many of the 20 per-core CPU% values exceed
        # MIMESYS_DENSITY_ACTIVE_THRESH (default 20). collated_trace[0..19] are
        # raw per-core CPU% values.
        _min_active = int(os.environ.get("MIMESYS_DENSITY_MIN_ACTIVE_CORES", "0"))
        _active_thresh = float(os.environ.get("MIMESYS_DENSITY_ACTIVE_THRESH", "20"))
        if _min_active > 0:
            n_before = len(data)
            def _active_count(d):
                raw = d["info"]["collated_trace"]
                return sum(1 for v in raw[:20] if v > _active_thresh)
            data = [d for d in data if _active_count(d) >= _min_active]
            print(f"[density-filter] kept {len(data)}/{n_before} samples "
                  f"(min_active_cores={_min_active} at >{_active_thresh}%)")

        random.shuffle(data)
        val_data   = data[: len(data) // 10]
        train_data = data[len(data) // 10 :]

        # ── Tier 1.2: importance weights for WeightedRandomSampler ────────────
        # MIMESYS_DENSITY_TARGET_PCMEAN (default 0, no weighting): bias sampling
        # toward per-core CPU means near this target (e.g. 42.8 for test).
        # weight_i = exp(-(per_core_mean_i - target)^2 / (2 * sigma^2)).
        _target_pcmean = float(os.environ.get("MIMESYS_DENSITY_TARGET_PCMEAN", "0"))
        _target_sigma  = float(os.environ.get("MIMESYS_DENSITY_TARGET_SIGMA",  "10"))
        self._train_weights = None
        if _target_pcmean > 0:
            weights = []
            for d in train_data:
                raw = d["info"]["collated_trace"]
                pc_mean = float(np.mean(raw[:20]))
                w = float(np.exp(-((pc_mean - _target_pcmean) ** 2) / (2 * _target_sigma ** 2)))
                weights.append(max(w, 1e-6))
            self._train_weights = weights
            print(f"[density-weight] importance weights computed: target_pcmean={_target_pcmean}, "
                  f"sigma={_target_sigma}, min/max={min(weights):.4f}/{max(weights):.4f}")

        if use_rl and rl_sample_method == "kmean":
            # Per-second loading + K-means medoid selection (K=3000 by default).
            test_data = self._load_test_data_per_second(test_data_path, metrics_range_dict)
            test_data = self._sample_rl_test_data_kmean(test_data, K=rl_kmean_k)
        else:
            test_data = self._load_test_data(test_data_path, metrics_range_dict)
            if use_rl:
                test_data = self._sample_rl_test_data(test_data)

        print(f"{len(test_data)} test samples before augmentation, {len(train_data)} train samples")
        if use_rl and not rl_use_high_resource_supplements:
            print(f"[rl] high-resource supplements DISABLED — using {len(test_data)} test samples only.")
        if use_rl and rl_use_high_resource_supplements:
            # For RL: supplement with high-resource training samples stratified across
            # all four resource classes (IO, CPU, LLC, BW). Use the RAW values in
            # info["collated_trace"] — `clean_trace` is normalized to [-1, 1] so the
            # raw-scale thresholds (io>1000, cpu>50, etc.) would never match.
            # Layout: [0..19]=per-core CPU%, [20]=io, [21]=l3_cache_usage (agg),
            #         [22]=memory_bandwidth (agg).
            IO_IDX  = 20
            LLC_IDX = 21
            BW_IDX  = 22

            def _raw(d): return d["info"]["collated_trace"]

            high_io_train  = [d for d in train_data if _raw(d)[IO_IDX] > io_raw_threshold]
            high_cpu_train = [d for d in train_data if float(np.mean(_raw(d)[:20])) > cpu_avg_threshold]
            high_llc_train = [d for d in train_data if _raw(d)[LLC_IDX] > llc_max_threshold]
            high_bw_train  = [d for d in train_data if _raw(d)[BW_IDX]  > bw_max_threshold]

            # Supplement budget = rl_train_ratio × max(len(test_data), rl_kmean_k).
            # Falling back to rl_kmean_k lets us run RL with NO test conds at all
            # (point test_data_path at an empty dir) — useful for algorithm sanity
            # tests where you want pure-train conds.
            _budget_base = max(len(test_data), rl_kmean_k)
            n_total_train = int(_budget_base * rl_train_ratio)
            n_high_io  = min(len(high_io_train),  int(n_total_train * rl_high_io_fraction))
            n_high_cpu = min(len(high_cpu_train), int(n_total_train * rl_high_cpu_fraction))
            n_high_llc = min(len(high_llc_train), int(n_total_train * rl_high_llc_fraction))
            n_high_bw  = min(len(high_bw_train),  int(n_total_train * rl_high_bw_fraction))
            n_normal   = max(0, n_total_train - n_high_io - n_high_cpu - n_high_llc - n_high_bw)

            for lst in (high_io_train, high_cpu_train, high_llc_train, high_bw_train):
                random.shuffle(lst)
            normal_train = train_data[:n_normal]

            print(f"RL data: {len(test_data)} test  +  {n_normal} normal-train  +  "
                  f"{n_high_io}/{len(high_io_train)} high-IO (>{io_raw_threshold:.0f})  +  "
                  f"{n_high_cpu}/{len(high_cpu_train)} high-CPU (avg>{cpu_avg_threshold:.0f}%)  +  "
                  f"{n_high_llc}/{len(high_llc_train)} high-LLC (max>{llc_max_threshold:.0f})  +  "
                  f"{n_high_bw}/{len(high_bw_train)} high-BW (max>{bw_max_threshold:.0f})")

            test_data = (test_data + normal_train
                         + high_io_train[:n_high_io]
                         + high_cpu_train[:n_high_cpu]
                         + high_llc_train[:n_high_llc]
                         + high_bw_train[:n_high_bw])
        else:
            test_data = test_data + train_data[: len(test_data) // 3]

        label_shape = torch.tensor(val_data[0]["label"], dtype=torch.float32).shape

        if use_rl:
            random.shuffle(test_data)
            rl_train = test_data[: len(test_data) // 10 * 8]
            rl_val = test_data[-32:]
            self.train_dataset = StressNgTestDataset(rl_train, label_shape, ddpo_batch_size)
            self.val_dataset   = StressNgTestDataset(rl_val,   label_shape, 256)
            self.test_dataset  = None
        else:
            print("Using supervised learning setting")
            self.train_dataset = StressNgDataset(train_data)
            self.val_dataset   = StressNgDataset(val_data)
            self.test_dataset  = StressNgTestDataset(test_data, label_shape)

        print(f"Trace range: {self.trace_range}")

    def _load_or_build_training_data(self, file_path: str, max_time_steps: int):
        cache_data  = Path(file_path) / "training_data.pkl"
        cache_range = Path(file_path) / "metrics_range_dict.pkl"

        if cache_data.exists() and cache_range.exists():
            print("Loading training data from cache.")
            with open(cache_data, "rb") as f:
                data = pickle.load(f)
            with open(cache_range, "rb") as f:
                metrics_range_dict = pickle.load(f)
            return data, metrics_range_dict

        print("Generating training data...")
        raw_dataset = []
        for r in range(self._NUM_ROUNDS):
            for i in range(self._NUM_CHUNKS):
                raw_dataset.extend(
                    read_metric_action_datasets_fleetbench(
                        file_path, r, i, samples_per_chunk=self._SAMPLES_PER_CHUNK
                    )
                )

        raw_dataset, _ = self._filter_and_aggregate(raw_dataset, max_time_steps)
        data, metrics_range_dict = get_datasets_from_metric_action_pairs(
            raw_dataset, max_time_steps=max_time_steps
        )

        with open(cache_data, "wb") as f:
            pickle.dump(data, f)
        with open(cache_range, "wb") as f:
            pickle.dump(metrics_range_dict, f)
        return data, metrics_range_dict

    def _filter_and_aggregate(self, raw_dataset: list, max_time_steps: int):
        med_metrics_array, std_metrics_array = [], []
        for metrics_output, actions in raw_dataset:
            num_slots = len(actions) + 1
            med_metrics: dict = defaultdict(lambda: defaultdict(float))
            std_metrics: dict = defaultdict(lambda: defaultdict(float))
            for target_metric, metrics in metrics_output.items():
                grouped: dict = defaultdict(list)
                for sample_idx, v in enumerate(metrics):
                    grouped[sample_idx % num_slots].append(v)
                # Always emit all num_slots slots (even if a short capture left some
                # empty) so med_metrics_array stays rectangular across samples. A
                # missing slot contributes 0 — same as an all-zero window.
                for slot_idx in range(num_slots):
                    group = grouped.get(slot_idx, [])
                    nonzero = [v for v in group if v != 0]
                    med_metrics[slot_idx][target_metric] = (
                        sorted(nonzero)[len(nonzero) // 2] if nonzero else 0
                    )
                    std_metrics[slot_idx][target_metric] = np.std(nonzero) if nonzero else 0
            med_metrics_array.append([[v for v in m.values()] for m in med_metrics.values()])
            std_metrics_array.append([[v for v in m.values()] for m in std_metrics.values()])

        med_arr = np.array(med_metrics_array)
        std_arr = np.array(std_metrics_array)
        bad_indices = set(filter_high_variance_data(med_arr, std_arr, threshold=0.1))

        filtered = []
        for sample_idx, (metrics_output, actions) in enumerate(raw_dataset):
            if sample_idx in bad_indices:
                continue
            metric_keys = list(metrics_output.keys())
            for metric_idx, target_metric in enumerate(metric_keys):
                metrics_output[target_metric] = [
                    med_arr[sample_idx][slot][metric_idx]
                    for slot in range(len(actions))
                ]
            filtered.append((metrics_output, actions))
        return filtered, {}

    def _load_test_data(self, test_data_path: str, metrics_range_dict: dict) -> list:
        test_data = []
        for file in Path(test_data_path).iterdir():
            if file.is_file():
                print("Processing test file:", file)
                test_metrics_list = read_metric_file_from_real_app(str(file))
                test_data.extend(get_val_datasets_from_metric_data(test_metrics_list, metrics_range_dict))
        return test_data

    def _load_test_data_per_second(self, test_data_path: str, metrics_range_dict: dict) -> list:
        """Whole-file per-second loading. Each test trace becomes N per-second samples
        (where N = file duration in seconds), with `prev_clean_trace` = the previous
        second's normalized metric (zeros for t=0). This matches the K=3000 RL design.
        """
        test_data = []
        for file in Path(test_data_path).iterdir():
            # Only stats-*.txt files are real test traces; pqos-*.log siblings are
            # loaded via the paired-path logic inside read_metric_file_per_second.
            if not (file.is_file() and file.name.startswith("stats-") and file.suffix == ".txt"):
                continue
            print("Processing test file (per-second):", file)
            test_metrics_list = read_metric_file_per_second(str(file))
            test_data.extend(get_val_datasets_from_metric_data(test_metrics_list, metrics_range_dict))
        return test_data

    def _load_test_data_per_second_indexed(self, test_data_path: str, metrics_range_dict: dict) -> list:
        """Like `_load_test_data_per_second` but also tags each per-second sample with
        the source `file_name` and `t_index` (its position within that file). This is
        required for the windowed-DTW path where the trainer needs to assemble 5-sec
        windows around K-means medoids."""
        all_data = []
        for file in Path(test_data_path).iterdir():
            # Same stats-only filter as _load_test_data_per_second — pqos-*.log are
            # paired via path replacement, not iterated as a top-level file.
            if not (file.is_file() and file.name.startswith("stats-") and file.suffix == ".txt"):
                continue
            print("Processing test file (per-second indexed):", file)
            test_metrics_list = read_metric_file_per_second(str(file))
            per_file = get_val_datasets_from_metric_data(test_metrics_list, metrics_range_dict)
            N = len(per_file)
            for t, item in enumerate(per_file):
                item["_file"] = file.name
                item["_t"]    = t
                item["_N"]    = N
            all_data.extend(per_file)
        return all_data

    @staticmethod
    def _expand_kmean_to_windows(per_file_index: dict, medoid_samples: list,
                                  window: int = 5) -> list:
        """For each K-means medoid sample, build a `window`-sec consecutive sequence
        centered on its `_t`. Returns dicts with:
          - "metric_seq":  (W, 23) normalized metric per second in the window
          - "prev_clean_trace": (23,) the metric BEFORE the window's first second
                                (zeros if window starts at t=0)
          - "raw_metric_seq_4d": (W, 4) raw [avg_cpu, io, llc, bw] for GT-side DTW
          - "_file", "_t_center", "_t_start", "_t_end", "_window": metadata
        Samples whose window would extend beyond a file boundary are SKIPPED (so all
        returned windows are fully in-file, length exactly W).
        """
        half = window // 2
        out = []
        for s in medoid_samples:
            fname = s["_file"]; t_center = s["_t"]; N = s["_N"]
            t_start = t_center - half
            t_end   = t_start + window - 1
            if t_start < 0 or t_end >= N:
                continue
            file_items = per_file_index[fname]
            seq_items = file_items[t_start : t_end + 1]
            metric_seq    = [it["clean_trace"]    for it in seq_items]            # (W, 23)
            raw_seq_4d    = [(sum(it["collated_trace"][:20]) / 20,
                              it["collated_trace"][-3],
                              it["collated_trace"][-2],
                              it["collated_trace"][-1])
                             for it in seq_items]                                  # (W, 4)  raw
            # Per-core CPU GT for per-core-CPU-DTW reward (Layout: collated_trace[:20] = per-core CPU%)
            raw_seq_percore = [list(it["collated_trace"][:20]) for it in seq_items]  # (W, 20) raw
            prev_first = file_items[t_start - 1]["clean_trace"] if t_start > 0 else [0.0] * 23
            out.append({
                "metric_seq":         metric_seq,
                "raw_metric_seq_4d":  raw_seq_4d,
                "raw_metric_seq_percore": raw_seq_percore,
                "prev_clean_trace":   prev_first,
                "_file":     fname,
                "_t_center": t_center,
                "_t_start":  t_start,
                "_t_end":    t_end,
                "_window":   window,
            })
        return out

    def _sample_rl_test_data_kmean_windowed(self, test_data_path: str,
                                             metrics_range_dict: dict,
                                             K: int = 3000, window: int = 5,
                                             random_state: int = 42) -> list:
        """K-means medoid selection on per-second samples, then expand each medoid into
        a `window`-second consecutive sequence. Returns up to K windowed dicts.
        The returned dicts are sequence-flavored (have `metric_seq`/`raw_metric_seq_4d`)
        and are NOT compatible with the standard StressNgTestDataset — the trainer's
        windowed-DTW path consumes them directly.
        """
        indexed = self._load_test_data_per_second_indexed(test_data_path, metrics_range_dict)
        # Build per-file index for fast lookup
        per_file = defaultdict(list)
        for s in indexed:
            per_file[s["_file"]].append(s)
        # Each file's list is already in t-order because get_val_datasets_from_metric_data emits t=0..N-1
        # but make sure by sorting:
        for f in per_file:
            per_file[f].sort(key=lambda x: x["_t"])

        # Apply K-means medoid selection on the FULL per-sec pool
        medoids = self._sample_rl_test_data_kmean(indexed, K=K, random_state=random_state)
        # Expand each medoid to a window
        windowed = self._expand_kmean_to_windows(dict(per_file), medoids, window=window)
        print(f"[rl-kmean-windowed] K={K} medoids → {len(windowed)} windows of length {window} "
              f"(dropped {len(medoids) - len(windowed)} for boundary)")
        return windowed

    @staticmethod
    def _sample_rl_test_data_kmean(test_data: list, K: int = 3000,
                                    random_state: int = 42) -> list:
        """K-means on per-second test points (4-D feature = avg-CPU, IO, LLC, BW from
        clean_trace), pick the medoid (sample closest to centroid) per cluster. Returns
        up to K representative samples. Matches the K=3000 RL selection scheme.
        """
        if len(test_data) <= K:
            print(f"[rl-kmean] only {len(test_data)} per-sec samples available (<= K={K}); "
                  f"returning all without clustering.")
            return list(test_data)
        feats = np.array([
            (
                sum(d["clean_trace"][:20]) / 20,  # avg CPU%
                d["clean_trace"][-3],             # io
                d["clean_trace"][-2],             # l3_cache_usage
                d["clean_trace"][-1],             # memory_bandwidth
            )
            for d in test_data
        ], dtype=np.float64)
        from sklearn.cluster import KMeans
        print(f"[rl-kmean] clustering {len(test_data)} per-sec test samples → K={K} medoids ...")
        km = KMeans(n_clusters=K, n_init=4, random_state=random_state).fit(feats)
        labels = km.labels_
        centers = km.cluster_centers_
        selected = []
        empty_clusters = 0
        for c in range(K):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                empty_clusters += 1
                continue
            dists = np.linalg.norm(feats[idxs] - centers[c], axis=1)
            medoid_local = idxs[int(np.argmin(dists))]
            selected.append(test_data[int(medoid_local)])
        print(f"[rl-kmean] selected {len(selected)} medoids (skipped {empty_clusters} empty clusters)")
        return selected

    @staticmethod
    def _sample_rl_test_data(test_data: list, K: int = 160000) -> list:
        sample_metrics = [
            (
                sum(d["clean_trace"][:20]) / 20,
                d["clean_trace"][-3],
                d["clean_trace"][-2],
                d["clean_trace"][-1],
            )
            for d in test_data
        ]
        bounds = [(min(m), max(m)) for m in zip(*sample_metrics)]
        n = max(1, int((K + 0.1) ** (1 / 4)))
        grid_axes = [np.linspace(lo, hi, n) for lo, hi in bounds]
        grid_points = np.array(np.meshgrid(*grid_axes, indexing="ij")).reshape(4, -1).T[:K]

        sample_arr = np.array(sample_metrics)
        batch_size = 1000
        sampled_indices: set = set()
        for start in range(0, len(grid_points), batch_size):
            batch = grid_points[start : start + batch_size]
            dists = np.linalg.norm(sample_arr[:, None, :] - batch[None, :, :], axis=2)
            sampled_indices.update(np.argmin(dists, axis=0).tolist())
        return [test_data[i] for i in list(sampled_indices)[:K]]

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        # If Tier 1.2 importance weights were computed, use WeightedRandomSampler
        # (replaces `shuffle=True`). num_samples=len(weights) preserves epoch size.
        if getattr(self, "_train_weights", None):
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(self._train_weights,
                                             num_samples=len(self._train_weights),
                                             replacement=True)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def test_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Benchmark test helper
# ---------------------------------------------------------------------------

class BenchmarkTestDataSet:
    def __init__(
        self,
        training_data_path: str,
        benchmark_base_path: str,
        benchmark_names: list[str],
    ):
        self.benchmark_base_path = benchmark_base_path
        self.benchmark_names = benchmark_names
        self.test_batch_size = 1

        cache_data  = Path(training_data_path) / "training_data.pkl"
        cache_range = Path(training_data_path) / "metrics_range_dict.pkl"
        metrics_range_dict = {}
        if cache_data.exists() and cache_range.exists():
            print("Loading training data from cache.")
            with open(cache_data, "rb") as f:
                data = pickle.load(f)
            with open(cache_range, "rb") as f:
                metrics_range_dict = pickle.load(f)

        label_shape = torch.tensor(data[0]["label"], dtype=torch.float32).shape
        test_metrics_list = self._get_traces()
        test_data = get_val_datasets_from_metric_data(test_metrics_list, metrics_range_dict)
        self.test_dataset = StressNgTestDataset(test_data, label_shape)

    def _get_traces(self) -> list:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(tqdm(
                executor.map(
                    lambda name: get_tacc_stats_and_energy_from_benchmark_name(
                        base_path=self.benchmark_base_path, benchmark_name=name
                    ),
                    self.benchmark_names,
                ),
                total=len(self.benchmark_names),
            ))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
