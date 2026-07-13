"""Data-format parsers for kernmlops (parquet), TACC stats, and PCM traces.

Each section is clearly delimited.  Functions that exist in multiple formats
are disambiguated with a ``_pcm`` suffix for the PCM variants.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars
import polars as pl

from mimesys.schema.constants import METRIC_INTERVAL_TO_US, system_by_num_cores, system_specs


# ===========================================================================
# Section 1 – kernmlops (parquet)
# ===========================================================================

def read_parquet_dir(
    data_dir: Path | str, *, benchmark_name: str | None = None
) -> dict[str, polars.DataFrame]:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    kernmlops_dfs = dict[str, polars.DataFrame]()
    dataframe_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    for dataframe_dir in dataframe_dirs:
        dfs = [
            polars.read_parquet(x)
            for x in dataframe_dir.iterdir()
            if x.is_file()
            and x.suffix == ".parquet"
            and (benchmark_name is None or x.suffixes[-2] == f".{benchmark_name}")
        ]
        kernmlops_dfs[dataframe_dir.name] = polars.concat(dfs, how="diagonal_relaxed")
    return kernmlops_dfs


def _update_metrics_dict(merged_metrics: dict[int, list[float]], df: polars.DataFrame) -> None:
    for row_idx, (timestamp, value) in enumerate(df.iter_rows()):
        if row_idx == 0:
            continue
        merged_metrics[timestamp].append(value)


def to_time_series(
    df: polars.DataFrame,
    target_metric: str,
    uptime_metric: str,
    aggregate_method: str = "max",
    diff: bool = True,
) -> polars.DataFrame:
    df = df.with_columns([polars.col(uptime_metric) // METRIC_INTERVAL_TO_US])
    if aggregate_method == "max":
        df = df.group_by(uptime_metric).agg(polars.col(target_metric).max())
    elif aggregate_method == "avg":
        df = df.group_by(uptime_metric).agg(polars.col(target_metric).mean())
    df = df.sort(uptime_metric)
    if diff:
        df = df.with_columns([polars.col(target_metric).diff()])
    merged_metrics = defaultdict(list)
    _update_metrics_dict(merged_metrics, df)
    return merged_metrics


def merge_metrics_by_cpu(
    df: polars.DataFrame, target_metric: str, uptime_metric: str
) -> dict[int, float]:
    metrics_by_cpu = defaultdict(lambda: defaultdict(int))
    for cpu_id, target_metric_per_cpu in df.group_by("cpu"):
        if target_metric == "cpu_usage":
            merged_metrics = to_time_series(
                target_metric_per_cpu, target_metric, uptime_metric,
                aggregate_method="avg", diff=False,
            )
        else:
            merged_metrics = to_time_series(target_metric_per_cpu, target_metric, uptime_metric)
        metrics_by_cpu[cpu_id[0]] = merged_metrics

    merged_metrics = defaultdict(list)
    for timestamps in metrics_by_cpu.values():
        for timestamp, value in timestamps.items():
            merged_metrics[timestamp].extend(value)
    return dict(sorted(merged_metrics.items()))


def aggregate_metrics(
    metrics: dict[int, list[float]], target_metric: str, aggregate_method: str = "avg"
) -> dict[int, float]:
    if aggregate_method == "p50":
        aggregation_func = lambda x: sorted(x)[int(0.5 * len(x))]
    elif aggregate_method == "p90":
        aggregation_func = lambda x: sorted(x)[int(0.9 * len(x))]
    elif aggregate_method == "max":
        aggregation_func = lambda x: sorted(x)[-1]
    else:
        aggregation_func = lambda x: sum(x) / len(x)

    min_timestamp = min(metrics.keys())
    aggregated_metrics = {}
    for timestamp, values in metrics.items():
        if "cpu_cycles" in target_metric or "task_clock" in target_metric:
            for value_idx, v in enumerate(values):
                if v < 0:
                    values[value_idx] = v + 2**32
        aggregated_metrics[int(timestamp - min_timestamp)] = aggregation_func(values)
    return aggregated_metrics


# ===========================================================================
# Section 2 – TACC stats
# ===========================================================================

def parse_trace_file(file_path):
    """Parse a TACC stats file into (header, grouped_traces)."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, []

    paragraphs = content.strip().split("\n\n")
    if not paragraphs:
        return None, []

    header = tuple(line.strip() for line in paragraphs[0].split("\n") if line.strip())
    grouped_traces = []
    for paragraph in paragraphs[1:]:
        lines = paragraph.strip().split("\n")
        if not lines or not lines[0].strip():
            continue
        grouped_trace = defaultdict(list)
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if idx == 0:
                grouped_trace["timestamp"].append(line)
            else:
                key = line.split(None, 1)[0]
                grouped_trace[key].append(line)
        grouped_traces.append(grouped_trace)

    return header, grouped_traces


def parse_energy_trace(file_path):
    """Parse a power log file into a list of (timestamp, power) tuples."""
    try:
        with open(file_path, "r") as f:
            content = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    power_readings = []
    for line in content:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        power_readings.append((float(parts[0]), float(parts[1])))
    return power_readings


def get_timestamp(grouped_trace):
    return float(grouped_trace["timestamp"][0].split()[0])


# Trace-type fallbacks: across machine generations the IMC / LLC counter names
# differ (Haswell: intel_hsw_*, Broadwell: intel_bdw_*, Skylake: intel_skx_*).
_IMC_TRACE_TYPES = ("intel_hsw_imc", "intel_bdw_imc", "intel_skx_imc")
_LLC_TRACE_TYPES = ("intel_hsw_cbo", "intel_bdw_cbo", "intel_skx_cha")


def _select_traces(grouped_trace, candidate_keys):
    for key in candidate_keys:
        traces = grouped_trace.get(key)
        if traces:
            return traces
    return []


def detect_system_spec_from_trace(grouped_trace):
    """Pick the right system_specs entry by inspecting which uncore counter
    families are present.  c220g2 and c220g5 both expose 20 physical cores so
    falling back to system_by_num_cores alone misclassifies c220g5 as c220g2.
    """
    if "intel_skx_imc" in grouped_trace or "intel_skx_cha" in grouped_trace:
        return system_specs["c220g5"]
    if "intel_hsw_imc" in grouped_trace or "intel_hsw_cbo" in grouped_trace:
        num_cpu_cores = len(grouped_trace.get("cpu", []))
        return system_by_num_cores.get(num_cpu_cores, system_specs["c220g2"])
    num_cpu_cores = len(grouped_trace.get("cpu", []))
    return system_by_num_cores.get(num_cpu_cores)


def get_memory_bandwidth_from_grouped_trace(grouped_trace):
    total_bandwidth_bytes = 0
    timestamp = float(grouped_trace["timestamp"][0].split()[0])
    for channel in _select_traces(grouped_trace, _IMC_TRACE_TYPES):
        data = channel.split()
        if len(data) < 10:
            continue
        total_bandwidth_bytes += (int(data[6]) + int(data[7])) * 64
    return timestamp, total_bandwidth_bytes


def get_memory_bandwidth_from_grouped_trace_per_socket(grouped_trace):
    imc_traces = _select_traces(grouped_trace, _IMC_TRACE_TYPES)
    traces_by_socket = defaultdict(lambda: defaultdict(int))
    for channel in imc_traces:
        data = channel.split()
        if len(data) < 10:
            continue
        socket_id = data[1].split("/")[0]
        traces_by_socket[socket_id]["read"] += int(data[6]) * 64
        traces_by_socket[socket_id]["write"] += int(data[7]) * 64
    traces_by_socket = dict(sorted(traces_by_socket.items()))
    return [traces_by_socket[s]["read"] + traces_by_socket[s]["write"] for s in traces_by_socket]


def get_memory_bandwidth_read_write_per_socket(grouped_trace):
    """Same as above but returns ([read_bytes_per_socket], [write_bytes_per_socket])."""
    imc_traces = _select_traces(grouped_trace, _IMC_TRACE_TYPES)
    traces_by_socket = defaultdict(lambda: defaultdict(int))
    for channel in imc_traces:
        data = channel.split()
        if len(data) < 10:
            continue
        socket_id = data[1].split("/")[0]
        traces_by_socket[socket_id]["read"] += int(data[6]) * 64
        traces_by_socket[socket_id]["write"] += int(data[7]) * 64
    traces_by_socket = dict(sorted(traces_by_socket.items()))
    reads  = [traces_by_socket[s]["read"]  for s in traces_by_socket]
    writes = [traces_by_socket[s]["write"] for s in traces_by_socket]
    return reads, writes


def get_l3_cache_usage_from_grouped_trace(grouped_trace):
    total = 0
    timestamp = float(grouped_trace["timestamp"][0].split()[0])
    for channel in _select_traces(grouped_trace, _LLC_TRACE_TYPES):
        data = channel.split()
        total += sum(int(x) for x in data[-4:])
    return timestamp, total


def get_l3_cache_usage_from_grouped_trace_per_socket(grouped_trace):
    traces_by_socket = defaultdict(lambda: defaultdict(int))
    for channel in _select_traces(grouped_trace, _LLC_TRACE_TYPES):
        data = channel.split()
        socket_id = data[1].split("/")[0]
        traces_by_socket[socket_id]["l3_cache_usage"] += sum(int(x) for x in data[-4:])
    traces_by_socket = dict(sorted(traces_by_socket.items()))
    return [traces_by_socket[s]["l3_cache_usage"] for s in traces_by_socket]


def get_qpi_usage_from_grouped_trace(grouped_trace):
    return sum(
        sum(int(x) for x in channel.split()[-4:])
        for channel in grouped_trace.get("intel_hsw_qpi", [])
    )


def get_qpi_usage_from_grouped_trace_per_socket(grouped_trace):
    total_by_socket = defaultdict(int)
    for channel in grouped_trace.get("intel_hsw_qpi", []):
        data = channel.split()
        total_by_socket[data[1].split("/")[0]] += sum(int(x) for x in data[-4:])
    return total_by_socket


def get_pcie_usage_from_grouped_trace(grouped_trace):
    return sum(
        sum(int(x) for x in channel.split()[-4:])
        for channel in grouped_trace.get("intel_hsw_r2pci", [])
    )


def get_pcie_usage_from_grouped_trace_per_socket(grouped_trace):
    total_by_socket = defaultdict(int)
    for channel in grouped_trace.get("intel_hsw_r2pci", []):
        data = channel.split()
        total_by_socket[data[1].split("/")[0]] += sum(int(x) for x in data[-4:])
    return total_by_socket


def parse_cpu_trace_into_list(cpu_trace):
    entry = cpu_trace.split()
    return {
        "id":      int(entry[1]),
        "user":    int(entry[2]),
        "nice":    int(entry[3]),
        "system":  int(entry[4]),
        "idle":    int(entry[5]),
        "iowait":  int(entry[6]),
        "irq":     int(entry[7]),
        "softirq": int(entry[8]),
    }


def parse_mem_trace_into_list(mem_trace):
    entry = mem_trace.split()
    return {
        "id":         int(entry[1]),
        "total":      int(entry[2]),
        "free":       int(entry[3]),
        "used":       int(entry[4]),
        "active":     int(entry[5]),
        "inactive":   int(entry[6]),
        "dirty":      int(entry[7]),
        "writeback":  int(entry[8]),
        "file_pages": int(entry[9]),
    }


def parse_net_trace_into_list(net_trace):
    entry = net_trace.split()
    iface = entry[1]
    if "ib" in iface or "lo" in iface:
        return None
    return {"id": iface, "rx_bytes": int(entry[4]), "tx_bytes": int(entry[16])}


def calculate_cpu_utilization_from_list(cpu_data_list):
    utilization = []
    for cpu_trace in cpu_data_list:
        d = parse_cpu_trace_into_list(cpu_trace)
        total = d["user"] + d["nice"] + d["system"] + d["idle"] + d["iowait"] + d["irq"] + d["softirq"]
        non_idle = d["user"] + d["nice"] + d["system"] + d["irq"] + d["softirq"]
        utilization.append((int(d["id"]), total, non_idle))
    utilization.sort(key=lambda x: x[0])
    return utilization


def average_cpu_utilization(prev_cpu_utilizations, current_cpu_utilizations):
    avg = []
    for prev in prev_cpu_utilizations:
        cpu_id = prev[0]
        curr = next(c for c in current_cpu_utilizations if c[0] == cpu_id)
        total_time = curr[1] - prev[1]
        non_idle_time = curr[2] - prev[2]
        avg.append((non_idle_time / total_time * 100) if total_time > 0 else 0.0)
    return avg


def calculate_memory_utilization_from_list(memory_data_list):
    sum_total = sum_free = sum_file_pages = 0
    for mem_trace in memory_data_list:
        d = parse_mem_trace_into_list(mem_trace)
        sum_total += d["total"]
        sum_free += d["free"]
        sum_file_pages += d["file_pages"]
    return sum_total, sum_total - (sum_free + sum_file_pages)


def calculate_network_utilization_from_list(network_data_list):
    for net_trace in network_data_list:
        d = parse_net_trace_into_list(net_trace)
        if d is None:
            continue
        return d["rx_bytes"] + d["tx_bytes"]
    return 0


def num_sectors_to_kb(num_sectors: int, sector_size: int = 512) -> int:
    return int(num_sectors * sector_size / 1024)


def calculate_io_utilization_from_list(io_data_list):
    # Disk device naming varies across hosts (the workload SSD may be sda or
    # sdb), so sum both; the inactive device idles near zero.
    read_kb_sum = write_kb_sum = 0
    for io_trace in io_data_list:
        line = io_trace.split()
        if line[1] in ("sda", "sdb"):
            read_kb_sum += num_sectors_to_kb(int(line[4]))
            write_kb_sum += num_sectors_to_kb(int(line[8]))
    return read_kb_sum, write_kb_sum


def calculate_numa_miss_from_list(numa_data_list):
    numa_hit_total = numa_miss_total = 0
    for numa_trace in numa_data_list:
        line = numa_trace.split()
        if len(line) < 3:
            continue
        numa_hit_total += int(line[2])
        numa_miss_total += int(line[3])
    return numa_hit_total, numa_miss_total


def _pad_or_truncate(metrics: dict, target_len: int) -> None:
    for key in metrics:
        metrics[key] = metrics[key][:target_len]
        if len(metrics[key]) < target_len:
            metrics[key] += [0] * (target_len - len(metrics[key]))


def process_trace(parsed_traces, period, duration):
    """Aggregate system-wide metrics from TACC stats parsed traces."""
    if not parsed_traces:
        return []

    num_cpu_cores = len(parsed_traces[0]["cpu"])
    system_spec = detect_system_spec_from_trace(parsed_traces[0])

    prev = defaultdict(int)
    prev["timestamp"] = 0
    metrics = {
        "memory_bandwidths":   [],
        "l3_cache_usages":     [],
        "qpi_usages":          [],
        "pcie_usages":         [],
        "memory_utilizations": [],
        "net_utilizations":    [],
        "avg_cpu_utilizations":[],
        "read_kb_sums":        [],
        "write_kb_sums":       [],
        "numa_hits":           [],
        "numa_misses":         [],
    }

    for trace in parsed_traces:
        if not trace:
            continue
        timestamp, memory_bandwidth = get_memory_bandwidth_from_grouped_trace(trace)
        _, l3_cache_usage = get_l3_cache_usage_from_grouped_trace(trace)
        qpi_usage = get_qpi_usage_from_grouped_trace(trace)
        pcie_usage = get_pcie_usage_from_grouped_trace(trace)
        cpu_utilization = calculate_cpu_utilization_from_list(trace["cpu"])
        mem_utilization = calculate_memory_utilization_from_list(trace["mem"])
        net_utilization = calculate_network_utilization_from_list(trace["net"])
        read_kb_sum, write_kb_sum = calculate_io_utilization_from_list(trace["block"])
        numa_hit, numa_miss = calculate_numa_miss_from_list(trace["numa"])

        if prev["timestamp"] == 0:
            prev.update({
                "timestamp": timestamp, "memory_bandwidth": memory_bandwidth,
                "l3_cache_usage": l3_cache_usage, "qpi_usage": qpi_usage,
                "pcie_usage": pcie_usage, "net_utilization": net_utilization,
                "cpu_utilization": cpu_utilization, "read_kb_sum": read_kb_sum,
                "write_kb_sum": write_kb_sum, "numa_hit": numa_hit, "numa_miss": numa_miss,
            })
            continue

        delta_time = timestamp - prev["timestamp"]
        if delta_time <= period:
            continue

        bandwidth_diff = (memory_bandwidth - prev["memory_bandwidth"]) / 1e9
        l3_cache_diff  = (l3_cache_usage  - prev["l3_cache_usage"])  / 1e6
        qpi_diff       = (qpi_usage        - prev["qpi_usage"])       / 1e6
        pcie_diff      = (pcie_usage       - prev["pcie_usage"])      / 1e6
        cpu_util_per_core = average_cpu_utilization(prev["cpu_utilization"], cpu_utilization)

        metrics["memory_bandwidths"].append(
            min(100, bandwidth_diff / delta_time / float(system_spec["Memory BW"]) * 100)
        )
        metrics["l3_cache_usages"].append(l3_cache_diff / delta_time)
        metrics["qpi_usages"].append(qpi_diff / delta_time)
        metrics["pcie_usages"].append(pcie_diff / delta_time)
        metrics["memory_utilizations"].append(mem_utilization[1] / mem_utilization[0] * 100)
        metrics["net_utilizations"].append(
            (net_utilization - prev["net_utilization"]) * 8 / 1e9 / delta_time
            / float(system_spec["Network"] * 100)
        )
        metrics["avg_cpu_utilizations"].append(
            min(150, sum(cpu_util_per_core) / len(cpu_util_per_core) * num_cpu_cores / int(system_spec["CPU"]))
        )
        metrics["read_kb_sums"].append((read_kb_sum  - prev["read_kb_sum"])  / delta_time)
        metrics["write_kb_sums"].append((write_kb_sum - prev["write_kb_sum"]) / delta_time)
        metrics["numa_hits"].append((numa_hit  - prev["numa_hit"])  / delta_time)
        metrics["numa_misses"].append((numa_miss - prev["numa_miss"]) / delta_time)

        prev.update({
            "timestamp": timestamp, "memory_bandwidth": memory_bandwidth,
            "l3_cache_usage": l3_cache_usage, "qpi_usage": qpi_usage,
            "pcie_usage": pcie_usage, "net_utilization": net_utilization,
            "cpu_utilization": cpu_utilization, "read_kb_sum": read_kb_sum,
            "write_kb_sum": write_kb_sum, "numa_hit": numa_hit, "numa_miss": numa_miss,
        })

    if duration > 0:
        _pad_or_truncate(metrics, int(duration / period))

    return {
        "cpu_util":       metrics["avg_cpu_utilizations"],
        "memory_bw":      metrics["memory_bandwidths"],
        "io_rw":          [r + w for r, w in zip(metrics["read_kb_sums"], metrics["write_kb_sums"])],
        "io_read":        metrics["read_kb_sums"],
        "io_write":       metrics["write_kb_sums"],
        "l3_cache_usage": metrics["l3_cache_usages"],
    }


def process_trace_granular(
    parsed_traces,
    period: float | None = None,
    duration: float = -1,
    include_aggregated_cpu: bool = False,
    aggregate_by_time: bool = False,
):
    """Per-socket / per-core granular metrics from TACC stats traces."""
    if not parsed_traces:
        return None

    num_cpu_cores = len(parsed_traces[0]["cpu"])
    system_spec = detect_system_spec_from_trace(parsed_traces[0])

    prev = defaultdict(int)
    metrics = defaultdict(list)
    epsilon = 0.1

    for trace in parsed_traces:
        if not trace:
            continue
        timestamp = get_timestamp(trace)
        bw_reads_per_socket, bw_writes_per_socket = \
            get_memory_bandwidth_read_write_per_socket(trace)
        try:
            l3_cache_usages = get_l3_cache_usage_from_grouped_trace_per_socket(trace)
        except ValueError as e:
            print(f"Error parsing L3 cache trace at timestamp {timestamp}: {e}")
            continue
        qpi_usage = get_qpi_usage_from_grouped_trace_per_socket(trace)
        pcie_usage = get_pcie_usage_from_grouped_trace_per_socket(trace)
        try:
            cpu_utilization = calculate_cpu_utilization_from_list(trace["cpu"])
        except IndexError as e:
            print(f"Error parsing CPU trace at timestamp {timestamp}: {e}")
            continue
        read_kb_sum, write_kb_sum = calculate_io_utilization_from_list(trace["block"])

        # Aggregate across sockets — per-socket attribution is unreliable on
        # dual-socket cache-resident workloads (CHA winner-takes-most artifact).
        memory_bandwidth_read_total  = sum(bw_reads_per_socket)
        memory_bandwidth_write_total = sum(bw_writes_per_socket)
        l3_cache_usage_total = sum(l3_cache_usages)

        if prev["timestamp"] == 0:
            prev.update({
                "timestamp": timestamp,
                "memory_bandwidth_read":  memory_bandwidth_read_total,
                "memory_bandwidth_write": memory_bandwidth_write_total,
                "l3_cache_usage":   l3_cache_usage_total,
                "qpi_usage": qpi_usage,
                "pcie_usage": pcie_usage,
                "cpu_utilization": cpu_utilization,
                "read_kb_sum": read_kb_sum,
                "write_kb_sum": write_kb_sum,
            })
            continue

        delta_time = timestamp - prev["timestamp"]
        if period is not None and delta_time <= period - epsilon:
            continue

        # read/write come from separate iMC counters; combined
        # `memory_bandwidth` = read+write is kept for backwards compat.
        bw_read_diff  = (memory_bandwidth_read_total  - prev["memory_bandwidth_read"])  / 1e9
        bw_write_diff = (memory_bandwidth_write_total - prev["memory_bandwidth_write"]) / 1e9
        peak_bw = float(system_spec["Memory BW"])
        bw_read_pct  = bw_read_diff  / delta_time / peak_bw * 100
        bw_write_pct = bw_write_diff / delta_time / peak_bw * 100
        metrics["memory_bandwidth_read"].append(bw_read_pct)
        metrics["memory_bandwidth_write"].append(bw_write_pct)
        metrics["memory_bandwidth"].append(bw_read_pct + bw_write_pct)

        l3_cache_diff = (l3_cache_usage_total - prev["l3_cache_usage"]) / 1e6
        metrics["l3_cache_usage"].append(l3_cache_diff / delta_time)

        cpu_util_per_core = average_cpu_utilization(prev["cpu_utilization"], cpu_utilization)
        for cpu_id, util in enumerate(cpu_util_per_core):
            metrics[f"avg_cpu_utilizations_core_{cpu_id:02d}"].append(util)

        io_read_kbps  = (read_kb_sum  - prev["read_kb_sum"])  / delta_time
        io_write_kbps = (write_kb_sum - prev["write_kb_sum"]) / delta_time
        metrics["io_read"].append(io_read_kbps)
        metrics["io_write"].append(io_write_kbps)
        metrics["io"].append(io_read_kbps + io_write_kbps)

        prev.update({
            "timestamp": timestamp,
            "memory_bandwidth_read":  memory_bandwidth_read_total,
            "memory_bandwidth_write": memory_bandwidth_write_total,
            "l3_cache_usage":   l3_cache_usage_total,
            "qpi_usage": qpi_usage,
            "pcie_usage": pcie_usage,
            "cpu_utilization": cpu_utilization,
            "read_kb_sum": read_kb_sum,
            "write_kb_sum": write_kb_sum,
        })

    if not metrics:
        return None
    if duration > 0 and period is not None:
        _pad_or_truncate(metrics, int(duration / period))
    if include_aggregated_cpu:
        cpu_util_keys = [k for k in metrics if k.startswith("avg_cpu_utilizations_core_")]
        metrics["avg_cpu_utilizations_total"] = [
            sum(vals) / len(vals)
            for vals in zip(*(metrics[k] for k in cpu_util_keys))
        ]
    if aggregate_by_time:
        metrics = {k: [sorted(v)[len(v) // 2]] for k, v in metrics.items()}

    return dict(sorted(metrics.items()))


def process_trace_all(parsed_traces, include_aggregated_cpu=False, aggregate_by_time=False):
    return process_trace_granular(
        parsed_traces, period=None,
        include_aggregated_cpu=include_aggregated_cpu,
        aggregate_by_time=aggregate_by_time,
    )


def process_trace_fine_grained(parsed_traces, period, duration, include_aggregated_cpu=False, aggregate_by_time=False):
    return process_trace_granular(
        parsed_traces, period=period, duration=duration,
        include_aggregated_cpu=include_aggregated_cpu,
        aggregate_by_time=aggregate_by_time,
    )


def get_energy_usage_time_series(energy_traces, period=1):
    if not energy_traces:
        return []
    time_series = []
    prev_timestamp = energy_traces[0][0]
    prev_power = None
    for timestamp, power in energy_traces:
        if prev_power is not None:
            if timestamp - prev_timestamp < period:
                continue
            prev_timestamp = timestamp
            delta = power - prev_power
            time_series.append(delta if delta > 0 else (time_series[-1] if time_series else 0))
        prev_power = power
    return time_series


def get_tacc_stats_and_energy_from_benchmark_name(
    base_path: str,
    benchmark_name: str,
    stats_file_format: str = "stats-{}.txt",
    energy_file_format: str = "power-{}.log",
    period: float = 2,
):
    tacc_stats_path = f"{base_path}/{stats_file_format.format(benchmark_name)}"
    energy_path = f"{base_path}/{energy_file_format.format(benchmark_name)}"
    _, parsed_traces = parse_trace_file(tacc_stats_path)
    energy_traces = parse_energy_trace(energy_path)
    metrics_output = process_trace_fine_grained(parsed_traces, period=period, duration=-1)
    get_energy_usage_time_series(energy_traces, period=period)
    return metrics_output


def aggregate_profiled_metrics(profiled_metrics: Dict[str, List]):
    if not profiled_metrics:
        return {}
    result_median, result_std = {}, {}
    for metric_name, values in profiled_metrics.items():
        if not values:
            continue
        arr = np.array(values)
        result_median[metric_name] = np.median(arr)
        result_std[metric_name] = np.std(arr)
    return result_median, result_std
