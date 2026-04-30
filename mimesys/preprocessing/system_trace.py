"""Functions to load and normalize system traces."""

def get_min_max(
    time_series_datasets: list[dict], target_metric: str
) -> tuple[float, float]:
    min_val = float("inf")
    max_val = float("-inf")
    for metric_list in time_series_datasets[target_metric]:
        min_val = min(min_val, min(metric_list))
        max_val = max(max_val, max(metric_list))
    return min_val, max_val


def normalize_trace(
    metrics_list: list[list[int]], min_val: float, max_val: float
) -> list[list[float]]:
    results = []
    for metrics in metrics_list:
        results.append([
            max(-1.0, (metric - min_val) / (max_val - min_val) * 2 - 1)
            if max_val > min_val else 0.0
            for metric in metrics
        ])
    return results


def unnormalize_trace(
    metrics_list: list[list[float]], min_val: float, max_val: float
) -> list[list[int]]:
    return [
        [int((metric + 1) / 2 * (max_val - min_val) + min_val) for metric in metrics]
        for metrics in metrics_list
    ]
