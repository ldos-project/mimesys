"""Functions to load and normalize system traces."""
import concurrent.futures
import os
import tempfile
import zipfile
from collections import defaultdict

from tqdm import tqdm

from mimesys.schema.constants import num_metrics_to_collect, perf_metric_pairs
from mimesys.preprocessing.parsers import (
    aggregate_metrics,
    merge_metrics_by_cpu,
    read_parquet_dir,
    to_time_series,
)


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


def get_time_series_metrics_from_parquet_table(
    row, df, start_idx=2, end_idx=3, trim=True
) -> dict[str, list[float]]:
    metrics_dict = defaultdict(dict)
    metrics_output = defaultdict(list)
    no_metrics = True

    for table_name, target_metric, merge_by_cpu in perf_metric_pairs:
        start, end = row[start_idx], row[end_idx]
        uptime_metric = "ts_uptime_us"
        target_df = df[table_name]
        target_df = target_df.filter(
            (target_df[uptime_metric] >= start) & (target_df[uptime_metric] <= end)
        )

        if merge_by_cpu:
            merged_metrics = merge_metrics_by_cpu(target_df, target_metric, uptime_metric)
            if len(merged_metrics.keys()) == 0:
                return None
        else:
            merged_metrics = to_time_series(target_df, target_metric, uptime_metric)
        aggregated_metrics = aggregate_metrics(merged_metrics, target_metric, "avg")

        num_metrics = (
            max(aggregated_metrics) - min(aggregated_metrics) + 1
            if not trim else num_metrics_to_collect
        )
        for timestamp, value in aggregated_metrics.items():
            if timestamp < num_metrics:
                if value >= 0:
                    metrics_dict[target_metric][timestamp] = value

        for timestamp in range(num_metrics):
            if timestamp not in metrics_dict[target_metric]:
                metrics_dict[target_metric][timestamp] = (
                    0 if timestamp == 0
                    else metrics_dict[target_metric][timestamp - 1]
                )

        metrics_output[target_metric] = [
            metrics_dict[target_metric][t] for t in range(1, num_metrics - 1)
        ]
        if no_metrics:
            no_metrics = all(value == 0 for value in metrics_output[target_metric])

    if no_metrics:
        return None
    return metrics_output


def read_dataframes_from_zipfiles(zip_file_path: str, recursive: bool = False) -> list:
    if os.path.isdir(zip_file_path):
        if recursive:
            val_zip_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(zip_file_path)
                for file in files if file.endswith(".zip")
            ]
        else:
            val_zip_files = [
                os.path.join(zip_file_path, f)
                for f in os.listdir(zip_file_path) if f.endswith(".zip")
            ]
    else:
        val_zip_files = [zip_file_path]

    def read_trace_file(zip_file):
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(tmpdirname)
                data_dir = f"{tmpdirname}/data"
                first_child_dir = next(os.scandir(data_dir)).path
                return read_parquet_dir(data_dir=first_child_dir)
            except Exception:
                print(f"Error reading zip file: {zip_file}")
                return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        dfs = list(tqdm(executor.map(read_trace_file, val_zip_files), total=len(val_zip_files)))
    return dfs


def get_time_series_metrics_from_zipfiles(zip_file_path: str) -> list[dict]:
    dfs = read_dataframes_from_zipfiles(zip_file_path)
    benchmark_table_name = "benchmark_run_info"
    val_metric_datasets = []

    for df in dfs:
        if df is None:
            continue

        def process_row(row):
            metrics_output = get_time_series_metrics_from_parquet_table(row, df, 1, 2, trim=False)
            return None if metrics_output is None else dict(metrics_output)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            val_metric_datasets.extend(list(tqdm(
                executor.map(process_row, list(df[benchmark_table_name].iter_rows())),
                total=len(df[benchmark_table_name]),
            )))

    chunked = []
    for metric_output in val_metric_datasets:
        num_values = min(len(v) for v in metric_output.values())
        for chunk_start in range(0, num_values, num_metrics_to_collect // 2):
            chunked_metric_output = {
                k: v[chunk_start + 1 : chunk_start + num_metrics_to_collect - 1]
                for k, v in metric_output.items()
                if len(v[chunk_start + 1 : chunk_start + num_metrics_to_collect - 1]) >= num_metrics_to_collect - 2
            }
            if chunked_metric_output:
                chunked.append(chunked_metric_output)
    return chunked


def get_time_series_metrics_from_test_results(test_file_path: str) -> list[dict]:
    dfs = read_dataframes_from_zipfiles(test_file_path)
    benchmark_table_name = "benchmark_run_info"
    metric_datasets = []

    for df in dfs:
        if df is None:
            continue

        def process_row(row):
            print(row)
            metrics_output = get_time_series_metrics_from_parquet_table(row, df)
            return None if metrics_output is None else dict(metrics_output)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            metric_datasets.extend(list(tqdm(
                executor.map(process_row, list(df[benchmark_table_name].iter_rows())),
                total=len(df[benchmark_table_name]),
            )))
    return metric_datasets


def tensor_to_metric_dict(metric_tensor):
    metric_list = sorted([e[1] for e in perf_metric_pairs])
    return {
        metric: metric_tensor.tolist()[
            i * (num_metrics_to_collect - 2) : (i + 1) * (num_metrics_to_collect - 2)
        ]
        for i, metric in enumerate(metric_list)
    }


def time_series_metrics_to_metric_vectors(
    metrics: dict[str, list[float]], aggregate_method: str = "avg"
):
    if aggregate_method == "p50":
        aggregation_func = lambda x: sorted(x)[int(0.5 * len(x))]
    elif aggregate_method == "p90":
        aggregation_func = lambda x: sorted(x)[int(0.9 * len(x))]
    elif aggregate_method == "max":
        aggregation_func = lambda x: sorted(x)[-1]
    else:
        aggregation_func = lambda x: sum(x) / len(x)

    return [
        aggregation_func(metrics[target_metric])
        for target_metric in sorted([e[1] for e in perf_metric_pairs])
        if target_metric in metrics
    ]
