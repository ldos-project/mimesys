from pydantic import BaseModel
import time
import os
import asyncio
import concurrent.futures
import tempfile

import paramiko

from mimesys.schema.machine import Machine
import shutil
import zipfile

from pytorch_lightning.loggers import Logger

from typing import Optional
import re
import numpy as np

from mimesys.preprocessing.parsers import parse_trace_file, process_trace_all
from mimesys.preprocessing.pqos_parser import pqos_metrics_dict, PQOS_METRIC_KEYS


def _merge_pqos_into_metrics(profiled_metrics, stats_path):
    """Load paired pqos.log next to stats and merge its 3 keys into profiled_metrics.
    Caller may pass any file_path; we replace 'stats' with 'pqos' and '.txt' with '.log'."""
    import os
    pqos_path = stats_path.replace("/stats-", "/pqos-").replace(".txt", ".log")
    if not os.path.exists(pqos_path):
        return profiled_metrics
    try:
        pq = pqos_metrics_dict(pqos_path)
    except Exception:
        return profiled_metrics
    n = len(next(iter(profiled_metrics.values()))) if profiled_metrics else 0
    for k in PQOS_METRIC_KEYS:
        v = pq.get(k, [])
        # Match the length of HPC metrics so EMD sees comparable arrays.
        if len(v) == n:
            profiled_metrics[k] = list(v)
        elif len(v) > 0:
            # Length mismatch: pad/truncate to n. Better than dropping.
            arr = list(v)[:n] + [0.0] * max(0, n - len(v))
            profiled_metrics[k] = arr
    return profiled_metrics

import json
from collections import defaultdict
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
    low_resource_penalty_weight: float = 0.0  # blend in relative L1 to penalize low-target errors more
    membw_reward_weight: float = 1.0  # multiplier on memory_bandwidth L1 reward term (socket-aggregated)
    llc_reward_weight: float = 1.0    # multiplier on l3_cache_usage L1 reward term (socket-aggregated)
    io_reward_weight: float = 1.0     # multiplier on io L1 reward term (applied in _parse_local_stats)
    cpu_reward_weight: float = 1.0    # multiplier on avg_cpu_utilizations* L1 reward terms
    # Per-metric measurement slice: keep entries where (m_idx % period == offset).
    # Defaults (3, 1) match the historical 2-window deploy ([prev_state, curr, sleep] cycle).
    # For 1-window deploys produced when train.ddpo.skip_prev_state=True, use (2, 0)
    # so the slice picks the `curr` entries of the [curr, sleep, curr, sleep, ...] trace.
    reward_slice_period: int = 3
    reward_slice_offset: int = 1
    # If True, the reward signal (`avg_emd_results_by_batch`) per sample is the
    # flat mean across all 23 metric keys (20 per-core CPU + io + llc + membw).
    # Per-core CPU dominates at 20/23 ≈ 87% of the reward. Default False uses
    # the 4-family-balanced weighted mean (cpu collapses to 1 family, ~25%).
    use_raw_mean_reward: bool = False

    class Config:
        arbitrary_types_allowed = True


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

            # Pull stats AND pqos files via SFTP. _merge_pqos_into_metrics
            # downstream needs the paired pqos-plan_*.log to inject the 3 libpqos
            # keys; without it, pqos EMD silently drops out of the reward signal.
            chunk_stats_dir = os.path.join(local_stats_dir, f"chunk_{host_idx}")
            os.makedirs(chunk_stats_dir, exist_ok=True)
            _, stdout, _ = client.exec_command(
                f"ls /users/{self.user_name}/results/stats-plan_*.txt "
                f"/users/{self.user_name}/results/pqos-plan_*.log 2>/dev/null"
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
            print(f"Chunk {host_idx}: pulled {count} stats+pqos files from {machine.hostname}")
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

        # Count plans we just uploaded so the remote can verify extraction matches.
        expected_plan_count = sum(1 for _ in os.listdir(plan_path) if _.endswith(".h5"))

        # Run the command on the remote machine
        print(f"Start running command on {machine.hostname} with {os.path.basename(file_path)} ({expected_plan_count} plans)")
        # Unzip the file and remove the zip file. Then verify extraction count matches.
        # Without the count check, a partial unzip + background script start can race so
        # the script sees only a handful of plans, silently runs them, and ships back
        # whatever stale stats files happen to be in ~/results from a previous run.
        command = (
            # Kill any leftover worker from a previous run and clear its stale
            # stats — otherwise a detached benchmark/collect process (e.g. after the
            # orchestrator was killed) races the new plans and stalls this chunk.
            # [x]-bracket regex so pkill doesn't match this dispatch shell's own
            # command line (which contains these patterns) and kill itself.
            f"sudo pkill -9 -f '[m]imesys_benchmark' 2>/dev/null; "
            f"pkill -9 -f '[c]ollect_mimesys_data' 2>/dev/null; "
            f"sudo pkill -9 -f '[b]azel.*mimesys' 2>/dev/null; "
            f"rm -rf /users/{self.user_name}/results/* 2>/dev/null; "
            f"cd {remote_execution_plan_path} && "
            f"rm -f *.h5 && "
            f"unzip -o {os.path.basename(file_path)} && "
            f"rm -f {os.path.basename(file_path)} && "
            f"actual=$(ls plan_*.h5 2>/dev/null | wc -l) && "
            f'if [ "$actual" -ne {expected_plan_count} ]; then '
            f'echo "PLAN_COUNT_MISMATCH expected={expected_plan_count} actual=$actual" >&2; exit 9; '
            f'fi'
        )
        machine.run_command(
            client=client,
            command=command,
        )

        print("Start collection command")
        mimesys_iters = os.environ.get("MIMESYS_ITERS", "4")
        # Forward MIMESYS_SLOT_US / MIMESYS_SLEEP from the controller env to the
        # worker. Worker's collect_mimesys_data.sh defaults SLOT_US=2000000 (2 s)
        # — set MIMESYS_SLOT_US=1000000 here for 1-s slots matching the windowed-
        # DTW deploy contract. Omitted vars fall back to worker defaults.
        env_prefix = f"MIMESYS_ITERS={mimesys_iters}"
        # MIMESYS_INTERNAL_PROFILING=1 enables the binary's hpcperfstatsd + pqos
        # collection. Default ON for training data collection so per-plan
        # stats-plan_NNN.txt AND pqos-plan_NNN.log are both produced.
        internal_prof = os.environ.get("MIMESYS_INTERNAL_PROFILING", "1")
        env_prefix += f" MIMESYS_INTERNAL_PROFILING={internal_prof}"
        slot_us = os.environ.get("MIMESYS_SLOT_US")
        if slot_us:
            env_prefix += f" MIMESYS_SLOT_US={slot_us}"
        sleep_env = os.environ.get("MIMESYS_SLEEP")
        if sleep_env is not None:
            env_prefix += f" MIMESYS_SLEEP={sleep_env}"
        machine.run_command_background(
            client=client,
            command=f"cd /users/{self.user_name} && {env_prefix} bash collect_mimesys_metrics.sh {self.user_name} {self.my_hostname} {destination_path} {host_idx} > collect_mimesys_metrics.log 2>&1"
        )

        print("Sent collection command")

        machine.close_connection(scp=scp, client=client)

    def get_found_files(self, chunk_indices, destination_path):
        """Return chunk_idx => zip filename for every chunk whose validation-*.zip
        has arrived at destination_path. Returns the dict only when all chunks
        are present, else None (kept for legacy callers that rely on this all-or-
        nothing semantics)."""
        partial = self._scan_found_files(chunk_indices, destination_path)
        if len(partial) == len(list(chunk_indices)):
            return partial
        return None

    def _scan_found_files(self, chunk_indices, destination_path):
        """Like get_found_files but returns whatever has arrived so far."""
        watch = {ci: f"validation-{ci}.zip" for ci in sorted(chunk_indices)}
        found = {}
        for root, _, files in os.walk(destination_path):
            for fname in files:
                for ci, suffix in watch.items():
                    if fname.endswith(suffix):
                        found[ci] = fname
        return dict(sorted(found.items()))

    def _count_local_plans(self, destination_path, chunk_idx):
        plans_dir = os.path.join(destination_path, f"chunk_{chunk_idx}", "plans")
        if not os.path.isdir(plans_dir):
            return 0
        return sum(1 for f in os.listdir(plans_dir) if f.endswith(".h5"))

    def _poll_remote_progress(self, pending_chunks):
        """SSH each pending chunk's worker and count stats-plan_*.txt files in
        /users/{user}/results — the in-progress signal that the benchmark is
        emitting per-plan stats. Returns {chunk_idx: count_or_-1}.

        Runs queries in parallel (one short-lived SSH per host) and swallows
        per-host errors as -1 so a single flaky host never silences the whole
        progress report."""
        if not pending_chunks:
            return {}

        def _query(ci):
            try:
                hostname = self.worker_host_names[ci]
            except IndexError:
                return ci, -1
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
                client.connect(hostname, username=self.user_name, pkey=key, timeout=10)
                _, stdout, _ = client.exec_command(
                    f"ls /users/{self.user_name}/results/stats-plan_*.txt 2>/dev/null | wc -l"
                )
                return ci, int(stdout.read().decode().strip() or "0")
            except Exception:
                return ci, -1
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(pending_chunks))) as ex:
            return dict(ex.map(_query, pending_chunks))

    async def wait_for_files(self, chunk_indices, destination_path,
                             timeout=60000, poll_interval=30):
        """Poll for validation-{idx}.zip per chunk. Prints a per-poll progress
        line so the user can see how many plans each worker has finished and
        which workers are still going."""
        chunks = sorted(chunk_indices)
        plan_totals = {ci: self._count_local_plans(destination_path, ci) for ci in chunks}
        start_time = time.time()
        round_label = os.path.basename(destination_path.rstrip("/")) or "round"
        first_done_ts = {}

        def _short(host):
            return host.split(".")[0]

        while True:
            elapsed = int(time.time() - start_time)
            arrived = self._scan_found_files(chunks, destination_path)
            done_now = set(arrived) - set(first_done_ts)
            for ci in done_now:
                first_done_ts[ci] = elapsed

            pending = [ci for ci in chunks if ci not in arrived]
            remote = self._poll_remote_progress(pending) if pending else {}

            bits = []
            for ci in chunks:
                host = (self.worker_host_names[ci]
                        if ci < len(self.worker_host_names) else f"chunk{ci}")
                tag = _short(host)
                tot = plan_totals.get(ci, "?")
                if ci in arrived:
                    bits.append(f"{tag}=DONE@{first_done_ts[ci]}s")
                else:
                    cnt = remote.get(ci, "?")
                    if cnt == -1:
                        bits.append(f"{tag}=ssh_err")
                    else:
                        bits.append(f"{tag}={cnt}/{tot}")
            print(f"  [{round_label} elapsed={elapsed:5d}s done={len(arrived)}/{len(chunks)}] "
                  + "  ".join(bits), flush=True)

            if len(arrived) == len(chunks):
                # legacy: extra sleep to let final scp/zip flush before parsing
                await asyncio.sleep(10)
                return arrived
            if time.time() - start_time > timeout:
                print(f"  [{round_label}] timeout after {timeout}s "
                      f"({len(arrived)}/{len(chunks)} done)", flush=True)
                return None
            await asyncio.sleep(poll_interval)

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
                    profiled_metrics = process_trace_all(parsed_traces, include_aggregated_cpu=True); profiled_metrics = _merge_pqos_into_metrics(profiled_metrics, file_path)
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

                    plan_stat_pairs.append(
                        (actions, avg_metrics_list, med_metrics_list, std_metrics_list)
                    )

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
                    profiled_metrics = process_trace_all(parsed_traces); profiled_metrics = _merge_pqos_into_metrics(profiled_metrics, file_path)

                    with open(ground_truth_path, "r") as f:
                        ground_truth = json.load(f)

                    with open(f"{ground_truth_path.rsplit('.')[0]}_predicted_{trial_idx}.json", "w") as f:
                        json.dump(profiled_metrics, f, indent=4)

                    emd_error = {}
                    metrics_std = {}
                    if profiled_metrics is None:
                        continue

                    metrics_weight = [1.0] * len(profiled_metrics)
                    for _wi, _name in enumerate(profiled_metrics.keys()):
                        if _name == "memory_bandwidth":
                            metrics_weight[_wi] = req.membw_reward_weight
                        elif _name == "l3_cache_usage":
                            metrics_weight[_wi] = req.llc_reward_weight

                    for metric_idx, (target_metric, metrics) in enumerate(profiled_metrics.items()):
                        if target_metric in ground_truth:
                            # emd = wasserstein_distance(metrics, ground_truth[target_metric])
                            m_min, m_max = trace_range.get(target_metric, (0, 0))

                            normalized_metrics = [(m - m_min) / (m_max - m_min + 1e-5) for m_idx, m in enumerate(metrics) if m_idx % req.reward_slice_period == req.reward_slice_offset]
                            # Drop curr_0 (warmup-contaminated first measured slot).
                            # Mirror the _parse_local_stats path; override with
                            # MIMESYS_KEEP_FIRST_CURR=1 for legacy compat.
                            if (os.environ.get("MIMESYS_KEEP_FIRST_CURR", "0") != "1"
                                    and len(normalized_metrics) > 1):
                                normalized_metrics = normalized_metrics[1:]
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
                    profiled_metrics = process_trace_all(parsed_traces); profiled_metrics = _merge_pqos_into_metrics(profiled_metrics, file_path)

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

                        # Keep entries at (m_idx % period == offset) — defaults
                        # (3, 1) match the legacy 2-window (prev, curr, sleep)×ITERS
                        # deploy. Override via ProfileRequest.reward_slice_period
                        # / .reward_slice_offset — e.g. (2, 0) for the 1-window
                        # (curr, sleep)×ITERS deploy emitted when skip_prev_state=True.
                        norm_pred = [
                            (m - m_min) / denom
                            for m_idx, m in enumerate(metrics)
                            if m_idx % req.reward_slice_period == req.reward_slice_offset
                        ]
                        # Drop curr_0 (warmup-contaminated first measured slot).
                        # Same default as preprocessing/dataloader.py — override
                        # with MIMESYS_KEEP_FIRST_CURR=1 for legacy compat.
                        if (os.environ.get("MIMESYS_KEEP_FIRST_CURR", "0") != "1"
                                and len(norm_pred) > 1):
                            norm_pred = norm_pred[1:]
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
                            emd_error[target_metric] = l1

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

        # Aggregate across batches; apply per-family reward weights from ProfileRequest
        # to the reward signal (avg_emd_results_by_batch). Keep merged_emd RAW so
        # callers (e.g. dynamic-λ logic in the trainer) can read unweighted per-metric
        # errors for adaptive weight updates.
        #
        # IMPORTANT: avg_cpu_utilizations is split into 4 sub-keys; if we treated each
        # as its own metric in the reward mean, cpu would silently get 4× weight vs
        # io/llc/bw. We aggregate cpu sub-keys to a single "cpu" family BEFORE applying
        # the weight, so the reward is a balanced mean of 4 resource families.
        def _family(name: str) -> str:
            if name.startswith("avg_cpu_utilizations"): return "cpu"
            return name
        def _family_weight(fam: str) -> float:
            if fam == "memory_bandwidth": return float(req.membw_reward_weight)
            if fam == "l3_cache_usage":   return float(req.llc_reward_weight)
            if fam == "io":                return float(req.io_reward_weight)
            if fam == "cpu":               return float(req.cpu_reward_weight)
            return 1.0
        merged_emd: dict = defaultdict(list)   # raw per-key (still includes cpu sub-keys)
        avg_emd_results_by_batch = []
        for emd_dict in emd_results:
            for k, v in emd_dict.items():
                merged_emd[k].append(v)
            if emd_dict:
                if req.use_raw_mean_reward:
                    # Per-sample RAW mean across all 23 metric keys (cpu sub-keys
                    # NOT collapsed). CPU dominates 20/23 ≈ 87% of the reward.
                    avg_emd_results_by_batch.append(
                        float(np.mean(list(emd_dict.values())))
                    )
                else:
                    # 1) collapse to per-family error (cpu sub-keys -> mean cpu)
                    fam_errs = defaultdict(list)
                    for m, err in emd_dict.items():
                        fam_errs[_family(m)].append(err)
                    fam_means = {f: float(np.mean(v)) for f, v in fam_errs.items()}
                    # 2) per-family weighted mean (now exactly 4 terms: io,llc,bw,cpu)
                    weighted = [_family_weight(f) * err for f, err in fam_means.items()]
                    avg_emd_results_by_batch.append(float(np.mean(weighted)))
            else:
                avg_emd_results_by_batch.append(0.0)

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
