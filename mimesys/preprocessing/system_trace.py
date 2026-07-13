"""Functions to load and normalize system traces.

Two normalization modes, selected via the ``MIMESYS_NORM_MODE`` env var:

  - "linear" (default): standard min-max → [-1, 1].
  - "log":    log1p(x - min) / log1p(max - min), then → [-1, 1]. Resolves the
              low-resource tail of heavily skewed metrics (LLC, BW, IO).
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
    ``MIMESYS_NORM_MODE``."""
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
