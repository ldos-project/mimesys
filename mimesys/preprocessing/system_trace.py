"""Functions to load and normalize system traces.

Two normalization modes are available, selected per-call:

  - "linear"  (default): standard min-max → [-1, 1].
  - "log":     compress the dynamic range first, then min-max → [-1, 1].
               Applies log1p((x - min)) and divides by log1p(max - min).
               Useful when the metric distribution is heavily skewed (LLC,
               BW, IO) and the model needs to resolve the low-resource tail
               that linear scaling crams into [-1, -0.9].

The active mode is read once from the env var ``MIMESYS_NORM_MODE`` so it
applies consistently to dataloader.py, engine.py, k3000_medoid_dataset.py and
any other caller without threading the choice through every API. To switch
modes for one process, export ``MIMESYS_NORM_MODE=log`` before launch.
"""
import math
import os


def _get_mode() -> str:
    """Read at call-time so test-time overrides take effect."""
    m = os.environ.get("MIMESYS_NORM_MODE", "linear").lower()
    if m not in ("linear", "log"):
        raise ValueError(f"MIMESYS_NORM_MODE must be 'linear' or 'log', got {m!r}")
    return m


def get_min_max(
    time_series_datasets: list, target_metric: str
) -> tuple[float, float]:
    min_val = float("inf")
    max_val = float("-inf")
    for metric_list in time_series_datasets[target_metric]:
        min_val = min(min_val, min(metric_list))
        max_val = max(max_val, max(metric_list))
    return min_val, max_val


def normalize_trace(
    metrics_list: list, min_val: float, max_val: float
) -> list:
    """Normalize a metric series to [-1, 1] using the mode selected by
    ``MIMESYS_NORM_MODE``. Output range is the same for both modes so model
    architectures don't change; the distribution within [-1, 1] differs."""
    mode = _get_mode()
    if max_val <= min_val:
        return [[0.0 for _ in row] for row in metrics_list]

    if mode == "linear":
        return [
            [max(-1.0, (m - min_val) / (max_val - min_val) * 2 - 1) for m in row]
            for row in metrics_list
        ]

    # log mode: log1p(x - min) / log1p(max - min) then [-1, 1].
    log_max = math.log1p(max_val - min_val)
    return [
        [max(-1.0, math.log1p(max(0.0, m - min_val)) / log_max * 2 - 1) for m in row]
        for row in metrics_list
    ]


def unnormalize_trace(
    metrics_list: list, min_val: float, max_val: float
) -> list:
    """Inverse of ``normalize_trace`` for the active mode."""
    mode = _get_mode()
    if max_val <= min_val:
        return [[int(min_val) for _ in row] for row in metrics_list]

    if mode == "linear":
        return [
            [int((m + 1) / 2 * (max_val - min_val) + min_val) for m in row]
            for row in metrics_list
        ]

    log_max = math.log1p(max_val - min_val)
    return [
        [int(math.expm1((m + 1) / 2 * log_max) + min_val) for m in row]
        for row in metrics_list
    ]
