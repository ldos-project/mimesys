from pydantic import BaseModel
import time
import os
import asyncio
import concurrent.futures
import tempfile

from mimesys.schema.machine import Machine
import shutil
import zipfile

from pytorch_lightning.loggers import Logger

from typing import Optional
import re
import numpy as np

from mimesys.preprocessing.parsers import parse_trace_file, process_trace, process_trace_fine_grained, aggregate_profiled_metrics, process_trace_all

from mimesys.schema.constants import max_time_steps
import json
from scipy.stats import wasserstein_distance
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import h5py
import random

class InitializeRequest(BaseModel):
    user_name: str
    private_key_path: str
    worker_host_names: list[str]
    my_hostname: str

class ProfileRequest(BaseModel):
    validation_data_path: str
    my_destination_path: str
    step: int
    num_batches: int
    num_trials: int
    logger: Optional[Logger] = None
    model_type: str = "ema"
    io_reward_weight: float = 1.0  # multiply IO metric's L1 loss by this weight in reward
    low_resource_penalty_weight: float = 0.0  # blend in relative L1 to penalize low-target errors more

    class Config:
        arbitrary_types_allowed = True

def visualize_system_traces(step: int, batch_idx: int, num_trials: int, validation_data_path: str, min_max_pkl: str = ""):
    ground_truth_path = f"{validation_data_path}/step_{step}_batch_{batch_idx}/system_traces.json"

    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)


    predicted_traces_list = []
    for trial_idx in range(num_trials):
        predicted_path = f"{validation_data_path}/step_{step}_batch_{batch_idx}/system_traces_predicted_{trial_idx}.json"
        with open(predicted_path, "r") as f:
            predicted_traces = json.load(f)
        predicted_traces_list.append(predicted_traces)


    # Here you can implement your visualization logic, e.g., using matplotlib or seaborn
    # For demonstration, we will just print the keys of the traces
    # print(f"Ground Truth Traces: {list(ground_truth.keys())}")
    # for trial_idx, predicted_traces in enumerate(predicted_traces_list):
    #     print(f"Predicted Traces for Trial {trial_idx}: {list(predicted_traces.keys())}")

    if min_max_pkl:
        with open(min_max_pkl, "rb") as f:
            min_max = pickle.load(f)

    trace_types = list(ground_truth.keys())
    num_traces = len(trace_types)

    # fig, axes = plt.subplots(num_traces, 1, figsize=(10, 4 * num_traces), squeeze=False)
    # axes = axes.flatten()

    gt_predicted_traces = defaultdict(list)
    for idx, trace_type in enumerate(trace_types):
        # hardcoded bugfix
        gt = ground_truth[trace_type]

        if min_max:
            min_val, max_val = min_max[trace_type]
            gt = [(metric - min_val) / (max_val - min_val) * 100 for metric in gt]

        gt_idx = idx

        # ax = axes[gt_idx]
        # ax.plot(gt, label="Ground Truth", linewidth=2, color="blue")

        # ax = axes[idx]
        for trial_idx, predicted_traces in enumerate(predicted_traces_list):
            if trace_type in predicted_traces:
                if min_max:
                    min_val, max_val = min_max[trace_type]
                    predicted_traces[trace_type] = [
                        (metric - min_val) / (max_val - min_val) * 100
                        for metric in predicted_traces[trace_type]
                    ]
                # ax.plot(predicted_traces[trace_type], label=f"Predicted {trial_idx}", linestyle="--", linewidth=2, color="orange")
        # ax.set_title(trace_type)
        # ax.legend()
        # ax.set_ylim(0, 100)
        # ax.set_xlabel("Time Step")
        # ax.set_ylabel("Value")

        gt_predicted_traces[idx].append(predicted_traces[trace_type])
        gt_predicted_traces[gt_idx].append(gt)


    emd_by_traces = defaultdict(float)
    for idx, traces in gt_predicted_traces.items():
        if len(traces) < 2:
            continue
        emd = wasserstein_distance(traces[0], traces[1])
        emd_by_traces[idx] = emd

    # plt.tight_layout()
    # plt.savefig(f"{validation_data_path}/step_{step}_batch_{batch_idx}/traces_predicted.png")

    return emd_by_traces


class Profiler:
    def __init__(self, init: InitializeRequest):
        # Add your initialization logic here
        self.user_name = init.user_name
        self.private_key_path = init.private_key_path
        self.worker_host_names = init.worker_host_names
        self.my_hostname = init.my_hostname

    def process_host(self, host_idx, machine: Machine, temp_dir: str, destination_path: str):
        """Legacy method kept for backward compatibility. Use _run_chunk_scatter instead."""
        self._run_chunk_scatter(host_idx, machine, temp_dir, destination_path)

    def _run_chunk_scatter(self, host_idx: int, machine: Machine, temp_dir: str, local_stats_dir: str) -> int:
        """
        Send a chunk of HDF5 execution plans to a remote machine, run the benchmark
        synchronously via collect_mimesys_data.sh, and pull stats files back.

        Returns the number of stats files pulled.
        """
        jitter = random.uniform(0, 5)
        print(f"Chunk {host_idx}: jitter {jitter:.2f}s before connecting to {machine.hostname}")
        time.sleep(jitter)

        chunk_dir = os.path.join(temp_dir, f"chunk_{host_idx}")
        if not os.path.isdir(chunk_dir):
            print(f"Chunk {host_idx}: directory missing, skipping")
            return 0
        plan_files = [f for f in os.listdir(chunk_dir) if f.endswith(".h5")]
        if not plan_files:
            print(f"Chunk {host_idx}: no plan files, skipping")
            return 0

        # Zip the chunk
        zip_path = os.path.join(temp_dir, f"chunk_{host_idx}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in plan_files:
                zf.write(os.path.join(chunk_dir, fname), fname)

        client, sftp = machine.initialize_connection(
            username=self.user_name,
            private_key_path=self.private_key_path,
        )
        try:
            # Upload zip
            remote_zip = f"/users/{self.user_name}/fleetbench/execution_plans/chunk_{host_idx}.zip"
            machine.file_transfer(scp=sftp, file_path=zip_path, destination=remote_zip)
            print(f"Chunk {host_idx}: zip transferred to {machine.hostname}")

            # Unzip on remote (overwrite existing .h5 files)
            machine.run_command(client=client, command=(
                f"cd /users/{self.user_name}/fleetbench/execution_plans/ "
                f"&& rm -f *.h5 "
                f"&& unzip -o chunk_{host_idx}.zip "
                f"&& rm chunk_{host_idx}.zip"
            ))

            # Run benchmark synchronously (waits for completion)
            print(f"Chunk {host_idx}: running benchmark on {machine.hostname} ({len(plan_files)} plans)...")
            machine.run_command(client=client, command=(
                f"cd /users/{self.user_name}/fleetbench && "
                f"bash collect_mimesys_data.sh > ~/benchmark.log 2>&1"
            ))
            print(f"Chunk {host_idx}: benchmark done on {machine.hostname}")

            # Pull stats files via SFTP
            chunk_stats_dir = os.path.join(local_stats_dir, f"chunk_{host_idx}")
            os.makedirs(chunk_stats_dir, exist_ok=True)
            _, stdout, _ = client.exec_command(
                f"ls /users/{self.user_name}/results/stats-plan_*.txt 2>/dev/null"
            )
            remote_files = stdout.read().decode().strip().splitlines()
            count = 0
            for remote_path in remote_files:
                remote_path = remote_path.strip()
                if not remote_path:
                    continue
                local_path = os.path.join(chunk_stats_dir, os.path.basename(remote_path))
                sftp.get(remote_path, local_path)
                count += 1
            print(f"Chunk {host_idx}: pulled {count} stats files from {machine.hostname}")
            return count
        finally:
            machine.close_connection(scp=sftp, client=client)

    def process_host_fleetbench(self, host_idx, machine: Machine, destination_path: str):
        print(f"Start distributing files to workers: {host_idx}")
        remote_execution_plan_path = f"/users/{self.user_name}/fleetbench/execution_plans"
        destination_path = f"{destination_path}/chunk_{host_idx}"
        # distribute the files to workers if they exist
        client, scp = machine.initialize_connection(
            username=self.user_name,
            private_key_path=self.private_key_path
        )

        # Zip the file_path directory before sending
        zip_filename = os.path.join(destination_path, f"chunk_{host_idx}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            plan_path = f"{destination_path}/plans"
            for root, _, files in os.walk(plan_path):
                for file in files:
                    file_full_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_full_path, plan_path)
                    zipf.write(file_full_path, arcname)
        file_path = zip_filename

        machine.file_transfer(
            scp=scp,
            file_path=file_path,
            destination=f"{remote_execution_plan_path}/{os.path.basename(file_path)}",
        )
        # os.remove(zip_filename)

        print(f"File {file_path} transferred to {machine.hostname}")

        # Run the command on the remote machine
        print(f"Start running command on {machine.hostname} with {os.path.basename(file_path)}")
        # Unzip the file and remove the zip file
        command=f"cd {remote_execution_plan_path} && rm *.h5 2>/dev/null || true && unzip -o {os.path.basename(file_path)} && rm {os.path.basename(file_path)}"
        machine.run_command(
            client=client,
            command=command,
        )

        print("Start collection command")
        machine.run_command_background(
            client=client,
            command=f"cd /users/{self.user_name} && bash collect_mimesys_metrics.sh {self.user_name} {self.my_hostname} {destination_path} {host_idx} > collect_mimesys_metrics.log 2>&1"
        )

        print("Sent collection command")

        machine.close_connection(scp=scp, client=client)

    def get_found_files(self, chunk_indices, destination_path):
        file_paths = []
        for chunk_idx in sorted(chunk_indices):
            file_paths.append(f"validation-{chunk_idx}.zip")

        found_files = {}
        dest_files = []
        for root, _, files in os.walk(destination_path):
            for file in files:
                dest_files.append(file)

        for fname in dest_files:
            for chunk_idx, watch_file in enumerate(file_paths):
                if fname.endswith(watch_file):
                    found_files[chunk_idx] = fname

        # Sort found_files by chunk_idx (key)
        found_files = {k: found_files[k] for k in sorted(found_files.keys())}

        if len(list(found_files.values())) == len(file_paths):
            return found_files
        return None

    async def wait_for_files(self, chunk_indices, destination_path, timeout=60000):
        # file_paths = []
        # for chunk_idx in sorted(chunk_indices):
        #     file_paths.append(f"validation-{chunk_idx}.zip")
        start_time = time.time()
        while True:
            # from "my_destination_path", iterate all files and check if file name suffix matches
            # file_path in file_paths.
            found_files = self.get_found_files(chunk_indices, destination_path)
            print(found_files)
            if found_files:
                await asyncio.sleep(10)
                return found_files
            if time.time() - start_time > timeout:
                return None
            await asyncio.sleep(10)

    def parse_metrics_from_zip(self, found_files: dict[int, str], destination_path: str, skip_parsing: bool = False):
        def get_plan_stat_pairs(chunk_idx, file_name):
            zip_file_name = f"{destination_path}/chunk_{chunk_idx}/{file_name}"
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(destination_path, f"chunk_{chunk_idx}"))

            if skip_parsing:
                return []

            plans_path = f"{destination_path}/chunk_{chunk_idx}/plans"
            stats_path = f"{destination_path}/chunk_{chunk_idx}/results"

            plan_stat_pairs = []
            for root, _, files in os.walk(plans_path):
                for file in files:
                    stat_fname = f"stats-{os.path.basename(file).split('.')[0]}.txt"
                    stat_fpath = f"{stats_path}/{stat_fname}"

                    header, parsed_traces = parse_trace_file(stat_fpath)
                    # profiled_metrics = process_trace_fine_grained(parsed_traces, period, duration, include_aggregated_cpu=True)
                    profiled_metrics = process_trace_all(parsed_traces, include_aggregated_cpu=True)
                    if not profiled_metrics:
                        continue

                    with h5py.File(f"{plans_path}/{os.path.basename(file)}", 'r') as f:
                        action_weights = f['execution_plan'][:]
                        actions = action_weights.tolist()

                    num_actions = len(actions) + 1

                    avg_metrics = defaultdict(lambda: defaultdict(float))
                    med_metrics = defaultdict(lambda: defaultdict(float))
                    std_metrics = defaultdict(lambda: defaultdict(float))
                    for target_metric, metrics in profiled_metrics.items():
                        grouped_metrics = defaultdict(list)
                        for metric_idx, v in enumerate(metrics):
                            group_idx = metric_idx % num_actions
                            grouped_metrics[group_idx].append(v)

                            for group_idx, group_metrics in grouped_metrics.items():
                                nonzero_values = [v for v in group_metrics if v != 0]
                                avg_value = sum(nonzero_values) / len(nonzero_values) if nonzero_values else 0
                                med_value = sorted(nonzero_values)[len(nonzero_values) // 2] if nonzero_values else 0
                                std_value = np.std(nonzero_values) if nonzero_values else 0
                                avg_metrics[group_idx][target_metric] = avg_value
                                med_metrics[group_idx][target_metric] = med_value
                                std_metrics[group_idx][target_metric] = std_value

                        # nonzero_values = [v for v in metrics if v != 0]
                        # avg_value = sum(nonzero_values) / len(nonzero_values) if nonzero_values else 0
                        # med_value = sorted(nonzero_values)[len(nonzero_values) // 2] if nonzero_values else 0
                        # std_value = np.std(nonzero_values) if nonzero_values else 0
                        # avg_metrics[target_metric] = avg_value
                        # med_metrics[target_metric] = med_value
                        # std_metrics[target_metric] = std_value

                    avg_metrics_list = [value for metrics in avg_metrics.values() for value in metrics.values()]
                    med_metrics_list = [value for metrics in med_metrics.values() for value in metrics.values()]
                    std_metrics_list = [value for metrics in std_metrics.values() for value in metrics.values()]

                    plan_stat_pairs.append((actions, avg_metrics_list, med_metrics_list, std_metrics_list))

            return plan_stat_pairs

        # Run compare_trace_files concurrently for all found_files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Wait for all compare_trace_files tasks to complete and collect results
            results = []
            if not found_files:
                return []
            for chunk_idx, file_name in found_files.items():
                future = executor.submit(get_plan_stat_pairs, chunk_idx, file_name)
                results.append(future)
            concurrent.futures.wait(results)
            plan_stat_pairs = [f.result() for f in results]

        plan_stat_pairs = [item for sublist in plan_stat_pairs for item in sublist]  # Flatten the list

        return plan_stat_pairs


    def on_files_created(
            self,
            found_files: dict[int, str],
            file_mapping_dict,
            req: ProfileRequest,
            trace_range: dict = {},
            period=2,
            duration=15,
            aggregate_time_series=False,
    ):
        def compare_trace_files(chunk_idx, file_name, aggregate_time_series=False):
            with tempfile.TemporaryDirectory() as unzip_dir:
                zip_path = os.path.join(req.my_destination_path, file_name)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)

                metrics_file_pair = []
                for root, _, files in os.walk(unzip_dir):
                    for file in files:
                        if not file.endswith("txt"):
                            continue
                        match = re.match(r"stats-plan_(\d+)\.txt", file)
                        if match:
                            file_idx = int(match.group(1))
                            file_path = os.path.join(root, file)

                            batch_idx, trial_idx = file_mapping_dict[(chunk_idx, file_idx)]
                            ground_truth_path = f"{req.validation_data_path}/step_{req.step}_batch_{batch_idx}/system_traces.json"

                            metrics_file_pair.append((batch_idx, trial_idx, file_path, ground_truth_path))

                metrics_file_pair.sort(key=lambda x: x[0])

                emd_errors = []
                metrics_std_list = []
                for batch_idx, trial_idx, file_path, ground_truth_path in metrics_file_pair:
                    header, parsed_traces = parse_trace_file(file_path)
                    profiled_metrics = process_trace_all(parsed_traces)

                    with open(ground_truth_path, "r") as f:
                        ground_truth = json.load(f)

                    with open(f"{ground_truth_path.rsplit('.')[0]}_predicted_{trial_idx}.json", "w") as f:
                        json.dump(profiled_metrics, f, indent=4)

                    emd_error = {}
                    metrics_std = {}
                    if profiled_metrics is None:
                        continue

                    metrics_weight = [1.0] * len(profiled_metrics)

                    for metric_idx, (target_metric, metrics) in enumerate(profiled_metrics.items()):
                        if target_metric in ground_truth:
                            # emd = wasserstein_distance(metrics, ground_truth[target_metric])
                            m_min, m_max = trace_range.get(target_metric, (0, 0))

                            normalized_metrics = [(m - m_min) / (m_max - m_min + 1e-5) for m_idx, m in enumerate(metrics) if m_idx % 3 == 1]
                            # normalized_metrics = [m for m in normalized_metrics if m > 0]
                            normalized_ground_truth = [(m - m_min) / (m_max - m_min + 1e-5) for m in ground_truth[target_metric]]

                            if aggregate_time_series:
                                normalized_metrics = np.array(normalized_metrics)
                                std_value = np.std(normalized_metrics, axis=0)
                                metrics_std[target_metric] = std_value

                                normalized_metrics = [np.median(normalized_metrics, axis=0)]

                            l1_loss = sum(abs(a - b) for a, b in zip(normalized_metrics, normalized_ground_truth)) / len(normalized_metrics)
                            mse = sum((a - b) ** 2 for a, b in zip(normalized_metrics, normalized_ground_truth)) / len(normalized_metrics)
                            relative_mse = sum(((a - b) / (b + 0.1)) ** 2 for a, b in zip(normalized_metrics, normalized_ground_truth)) / len(normalized_metrics)
                            # emd_errors[f"{target_metric}_mse"].append(mse)

                            emd_error[target_metric] = l1_loss * metrics_weight[metric_idx]

                            if target_metric == "memory_bandwidth":
                                print(f"{batch_idx},{file_path},{target_metric}: {ground_truth[target_metric]}, Predicted: {metrics}, MSE: {mse}, L1 Loss: {l1_loss}, weighted: {emd_error[target_metric]}")
                                print(f"Normalized GT: {normalized_ground_truth}, Normalized Predicted: {normalized_metrics}")
                            if target_metric == "avg_cpu_utilizations_core_00":
                                print(f"CPU00 - {batch_idx},{file_path},{target_metric}: {ground_truth[target_metric]}, Predicted: {metrics}, MSE: {mse}, L1 Loss: {l1_loss}, weighted: {emd_error[target_metric]}")

                    emd_errors.append(emd_error)
                    # merge standard deviation values
                    metrics_std_list.append(np.sqrt(np.mean(np.square(list(metrics_std.values())))))
                    # metrics_std_list.append(max(list(metrics_std.values())))

                # Remove the zip_path file after extraction
                os.remove(zip_path)

                return emd_errors, metrics_std_list

        # Wait for all compare_trace_files tasks to complete and collect results
        emd_results = []
        metrics_std_by_batch = []
        for chunk_idx, file_name in found_files.items():
            emd_errors, metrics_std = compare_trace_files(chunk_idx, file_name, aggregate_time_series=aggregate_time_series)
            emd_results.extend(emd_errors)
            metrics_std_by_batch.extend(metrics_std)

        merged_emd = defaultdict(list)
        avg_emd_results_by_batch = []
        for emd_dict in emd_results:
            for key, value in emd_dict.items():
                merged_emd[key].append(value)
            avg_emd_results_by_batch.append(np.mean(list(emd_dict.values())))

        avg_emd_by_metrics = {}
        for merged_key, merged_values in merged_emd.items():
            if merged_values:
                avg_emd = sum(merged_values) / len(merged_values)
                if req.logger:
                    req.logger.log_metrics({f"validation_emd_{merged_key}": avg_emd}, step=req.step)
                else:
                    print(f"validation_emd_{merged_key}: {avg_emd}")
                avg_emd_by_metrics[merged_key] = merged_values # avg_emd

        return avg_emd_by_metrics, avg_emd_results_by_batch, metrics_std_by_batch

    def _parse_local_stats(
        self,
        local_stats_dir: str,
        chunk_indices: set,
        file_mapping_dict: dict,
        req: "ProfileRequest",
        trace_range: dict,
        aggregate_time_series: bool,
    ):
        """
        Parse stats files pulled from remote machines into reward signals.

        Args:
            local_stats_dir: directory containing chunk_X/ subdirs with stats-plan_*.txt
            chunk_indices: set of chunk indices that were processed
            file_mapping_dict: (chunk_idx, file_idx) -> (batch_idx, trial_idx)
            req: ProfileRequest
            trace_range: per-metric (min, max) for normalization
            aggregate_time_series: if True, use median over time steps

        Returns:
            (avg_emd_by_metrics, avg_emd_results_by_batch, metrics_std_by_batch)
        """
        trace_range = trace_range or {}
        emd_results = []
        metrics_std_by_batch = []

        for chunk_idx in sorted(chunk_indices):
            chunk_dir = os.path.join(local_stats_dir, f"chunk_{chunk_idx}")
            if not os.path.isdir(chunk_dir):
                print(f"Warning: no stats directory for chunk {chunk_idx}")
                continue

            metrics_file_pair = []
            for fname in os.listdir(chunk_dir):
                match = re.match(r"stats-plan_(\d+)\.txt", fname)
                if not match:
                    continue
                file_idx = int(match.group(1))
                if (chunk_idx, file_idx) not in file_mapping_dict:
                    print(f"Warning: unmapped stats file chunk={chunk_idx} file={file_idx}")
                    continue
                batch_idx, trial_idx = file_mapping_dict[(chunk_idx, file_idx)]
                ground_truth_path = (
                    f"{req.validation_data_path}/step_{req.step}_batch_{batch_idx}/system_traces.json"
                )
                metrics_file_pair.append((batch_idx, trial_idx, os.path.join(chunk_dir, fname), ground_truth_path))

            metrics_file_pair.sort(key=lambda x: x[0])

            for batch_idx, trial_idx, file_path, ground_truth_path in metrics_file_pair:
                try:
                    header, parsed_traces = parse_trace_file(file_path)
                    profiled_metrics = process_trace_all(parsed_traces)

                    with open(ground_truth_path) as f:
                        ground_truth = json.load(f)

                    # Save predicted trace alongside ground truth
                    pred_path = f"{ground_truth_path.rsplit('.', 1)[0]}_predicted_{trial_idx}.json"
                    with open(pred_path, "w") as f:
                        json.dump(profiled_metrics, f, indent=4)

                    emd_error = {}
                    metrics_std = {}
                    if profiled_metrics is None:
                        continue

                    for target_metric, metrics in profiled_metrics.items():
                        if target_metric not in ground_truth:
                            continue
                        m_min, m_max = trace_range.get(target_metric, (0, 0))
                        denom = (m_max - m_min) + 1e-5

                        # idx % 3 == 1 corresponds to the "current action" segment
                        norm_pred = [
                            (m - m_min) / denom
                            for m_idx, m in enumerate(metrics) if m_idx % 3 == 1
                        ]
                        norm_gt = [(m - m_min) / denom for m in ground_truth[target_metric]]

                        if aggregate_time_series:
                            arr = np.array(norm_pred)
                            metrics_std[target_metric] = float(np.std(arr))
                            norm_pred = [float(np.median(arr))]

                        if norm_pred and norm_gt:
                            n = min(len(norm_pred), len(norm_gt))
                            l1 = sum(abs(a - b) for a, b in zip(norm_pred[:n], norm_gt[:n])) / n
                            # Relative L1: penalise errors more when the target is small
                            # (|pred-gt| / (gt + 0.1)) in normalised space
                            if req.low_resource_penalty_weight > 0.0:
                                rel_l1 = sum(
                                    abs(a - b) / (b + 0.1)
                                    for a, b in zip(norm_pred[:n], norm_gt[:n])
                                ) / n
                                w = req.low_resource_penalty_weight
                                l1 = (1.0 - w) * l1 + w * rel_l1
                            # Weight IO metric more heavily to encourage better IO prediction
                            weight = req.io_reward_weight if target_metric == "io" else 1.0
                            emd_error[target_metric] = l1 * weight

                            if target_metric == "memory_bandwidth":
                                print(f"batch={batch_idx} {target_metric}: gt={ground_truth[target_metric]} "
                                      f"pred={metrics} L1={l1:.4f}")

                    if req.logger and emd_error:
                        for k, v in emd_error.items():
                            req.logger.log_metrics({f"validation_emd_{k}": v}, step=req.step)
                    else:
                        for k, v in emd_error.items():
                            print(f"validation_emd_{k}: {v:.4f}")

                    emd_results.append(emd_error)
                    std_val = (
                        float(np.sqrt(np.mean(np.square(list(metrics_std.values())))))
                        if metrics_std else 0.0
                    )
                    metrics_std_by_batch.append(std_val)

                except Exception as e:
                    print(f"Error parsing chunk={chunk_idx} batch={batch_idx}: {e}")

        # Aggregate across batches
        merged_emd: dict = defaultdict(list)
        avg_emd_results_by_batch = []
        for emd_dict in emd_results:
            for k, v in emd_dict.items():
                merged_emd[k].append(v)
            avg_emd_results_by_batch.append(float(np.mean(list(emd_dict.values()))) if emd_dict else 0.0)

        avg_emd_by_metrics = dict(merged_emd)
        return avg_emd_by_metrics, avg_emd_results_by_batch, metrics_std_by_batch

    async def profile(self, req: ProfileRequest, trace_range: dict = None, period: int = 2, duration: int = 2, aggregate_time_series: bool = False):
        """
        Distribute execution plans to remote workers, run collect_mimesys_data.sh
        synchronously on each, pull stats files back, and compute reward signals.
        """
        machines = [Machine.from_hostname(hostname) for hostname in self.worker_host_names]
        chunk_indices = set()
        file_mapping_dict = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_size = (req.num_batches + len(machines) - 1) // len(machines)
            for batch_idx in range(req.num_batches):
                chunk_idx = batch_idx // chunk_size
                batch_path = f"{req.validation_data_path}/step_{req.step}_batch_{batch_idx}"
                os.makedirs(f"{temp_dir}/chunk_{chunk_idx}", exist_ok=True)
                chunk_indices.add(chunk_idx)

                for trial_idx in range(req.num_trials):
                    predicted_action_path = f"{batch_path}/predicted_actions_{req.model_type}_{trial_idx}.h5"
                    if os.path.exists(predicted_action_path):
                        file_idx = (batch_idx % chunk_size) * req.num_trials + trial_idx
                        dest_path = os.path.join(temp_dir, f"chunk_{chunk_idx}", f"plan_{file_idx:06d}.h5")
                        shutil.copy(predicted_action_path, dest_path)
                        file_mapping_dict[(chunk_idx, file_idx)] = (batch_idx, trial_idx)

            print(f"Copied files for {len(chunk_indices)} chunks. Distributing and running on {len(machines)} workers...")

            # Use a stable local directory for pulled stats (not inside temp_dir which gets deleted)
            local_stats_dir = os.path.join(
                req.my_destination_path, f"rl_stats_step{req.step}"
            )
            os.makedirs(local_stats_dir, exist_ok=True)

            expected = {i for i in range(len(machines)) if i * chunk_size < req.num_batches}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(machines)) as executor:
                futures = [
                    executor.submit(self._run_chunk_scatter, i, machines[i], temp_dir, local_stats_dir)
                    for i in expected
                ]
                concurrent.futures.wait(futures)

        avg_emd_by_metrics, avg_emd_results_by_batch, metrics_std_by_batch = self._parse_local_stats(
            local_stats_dir, chunk_indices, file_mapping_dict, req, trace_range, aggregate_time_series
        )
        return avg_emd_by_metrics, avg_emd_results_by_batch, metrics_std_by_batch

    async def profile_actions(self, destination_path, skip_parsing: bool = False):
        machines = [Machine.from_hostname(hostname) for hostname in self.worker_host_names]

        # Skip if files already exist
        found_files = self.get_found_files([i for i in range(len(machines))], destination_path)
        if found_files:
            return self.parse_metrics_from_zip(found_files, destination_path, skip_parsing)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_host_fleetbench, host_idx, hostname, destination_path)
                for host_idx, hostname in enumerate(machines)
            ]
            concurrent.futures.wait(futures)

        found_files = await self.wait_for_files([i for i in range(len(machines))], destination_path)
        plan_stat_pairs = self.parse_metrics_from_zip(found_files, destination_path, skip_parsing)
        return plan_stat_pairs

if __name__ == "__main__":
    emd_by_traces = defaultdict(list)
    for step in range(70400, 70472):
        for batch_idx in range(256):
            emd = visualize_system_traces(
                step=step,
                batch_idx=batch_idx,
                num_trials=1,
                validation_data_path="/home/dhkim/workspace/stress_emulate/llm-app-generation/mimesys/stress_ng_dataloader/train/diffusion/stress_ng_training_10k_bs_1024_vis_lr5e-4_tacc_stats_3_rl/training",
                min_max_pkl="/home/dhkim/tacc_stats_results/data/metrics_range_dict.pkl"
            )

            for trace_type, emd_value in emd.items():
                emd_by_traces[trace_type].append(emd_value)

        # Print the average EMD for each trace type
        avg_emd_sum = 0
        for trace_type, emd_values in emd_by_traces.items():
            if emd_values:
                avg_emd = sum(emd_values) / len(emd_values)
                avg_emd_sum += avg_emd
                print(f"Average EMD for {trace_type}: {avg_emd}")

        print(f"Total Average EMD for step {step}: {avg_emd_sum / len(emd_by_traces)}")
        with open("average_emd_results.txt", "a") as f:
            f.write(f"{avg_emd_sum / len(emd_by_traces)}\n")
        print("Profiler script executed successfully.")

    exit(0)

    # Prepare initialize parameters
    initialize_params = {
        "user_name": "dhkim",
        "private_key_path": "/home/dhkim/.ssh/id_rsa_utns",
        "worker_host_names": ["c220g2-010813.wisc.cloudlab.us"],
        "my_hostname": "mew3"
    }
    # Prepare profile parameters
    profile_params = {
        "validation_data_path": "/home/dhkim/workspace/stress_emulate/llm-app-generation/mimesys/stress_ng_dataloader/train/diffusion/stress_ng_training_10k_bs_1024_vis_lr5e-4_tacc_stats_2",
        "my_destination_path": "/home/dhkim/tacc_stats_results",
        "step": 35200,
        "num_batches": 10,
        "num_trials": 5,
    }

    # Initialize Profiler
    init_request = InitializeRequest(**initialize_params)
    profiler = Profiler(init_request)

    # Run async profile function
    async def main():
        profile_request = ProfileRequest(**profile_params)
        await profiler.profile(profile_request)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    pending = asyncio.all_tasks(loop=loop)
    pending = [task for task in pending if not task.done()]
    if pending:
        loop.run_until_complete(asyncio.gather(*pending))
