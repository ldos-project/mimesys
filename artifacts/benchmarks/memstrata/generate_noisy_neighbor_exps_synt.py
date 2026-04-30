from collections import defaultdict
import os
import subprocess
import time
from typing import Optional

webserver_list = [
    "dacapo_cassandra",
    "dacapo_kafka",
    "dacapo_spring",
    "dacapo_tomcat",
    "finagle_chirper",
    "finagle_http",
]

deathstartbench_list = [
    "deathstarbench_media",
    "deathstarbench_social",
]

spark_list = [
    "spark_als",
    "spark_gbt",
    "spark_lda",
    "spark_lr",
    "spark_pca",
    "spark_rf",
    "spark_sort",
    "spark_svd",
    "spark_terasort",
    "spark_wordcount",
]

database_list = [
    "silo_tpcc",
]

kvstore_list = [
    "redis_uniform_ycsb_a",
    "redis_ycsb_a",
    "redis_ycsb_b",
    "redis_ycsb_c",
    "redis_ycsb_d",
    "redis_ycsb_e",
    "redis_ycsb_f",
    "memcached_uniform_ycsb_a",
    "memcached_ycsb_a",
    "memcached_ycsb_b",
    "memcached_ycsb_c",
    "memcached_ycsb_d",
    "memcached_ycsb_e",
    "memcached_ycsb_f",
    "faster_uniform_ycsb_a",
    "faster_uniform_ycsb_b",
    "faster_uniform_ycsb_c",
    "faster_uniform_ycsb_f",
    "faster_ycsb_a",
    "faster_ycsb_b",
    "faster_ycsb_c",
    "faster_ycsb_f",
]

ml_list = [
    "stable_diffusion",
    "resnet50",
]

gapbs_list = [
    "bc-road",
    "bfs-road",
    "cc-road",
    "pr-road",
    "sssp-road",
    "tc-road",
]

target_app_lists = [
    "silo_tpcc",
    # "spark_terasort",
    # "resnet50",
    # "memcached_ycsb_a",
    # "redis_ycsb_a",
    # "faster_ycsb_a",
]

vms_per_workload = {
    "webserver": [1, 2],
    "deathstarbench": [3],
    "spark": [4],
    "database": [6],
    "kvstore": [7, 8],
    "ml": [9],
    "graph": [10],
}

class SyntheticWorkloadLauncher:
    def __init__(
            self,
            real_lock_path: str,
            synt_lock_path: str,
            workload_arg: str,
            vm_arg: str,
            start_core_idx: int,
            data_dir: str,
            target_app: str,
            idx: int,
            **kwargs,
            ):
        self.real_lock_path = real_lock_path
        self.synt_lock_path = synt_lock_path
        self.workload_arg = workload_arg
        self.vm_arg = vm_arg
        self.start_core_idx = start_core_idx
        self.data_dir = data_dir
        self.target_app = target_app
        self.idx = idx
        self.kwargs = kwargs

    def initialize_lock(self, lock_path: str):
        lock_dir = os.path.abspath(os.path.dirname(lock_path))
        for file in os.listdir(lock_dir):
            if file.endswith(".lock"):
                subprocess.run(["sudo", "rm", os.path.join(lock_dir, file)], check=True)
        subprocess.run(["sudo", "bash", "-c", f"echo 0 > {lock_path}"], check=True)

    def wait_for_lock_prepared(self, lock_path: str):
        while True:
            if os.path.exists(lock_path):
                with open(lock_path, "r") as f:
                    if f.read().strip() == "1":
                        break
                    time.sleep(1)
            else:
                raise FileNotFoundError(f"Synthetic lock file {lock_path} not found.")

    def write_lock_start(self, lock_path: str, value: str="2"):
        subprocess.run(["sudo", "bash", "-c", f"echo {value} > {lock_path}"], check=True)

    def prepare_synt_workloads(self):
        self.initialize_lock(self.synt_lock_path)

    def prepare_real_workloads(self):
        self.initialize_lock(self.real_lock_path)

    def wait_for_synt_prepared(self):
        self.wait_for_lock_prepared(self.synt_lock_path)

    def wait_for_real_prepared(self):
        self.wait_for_lock_prepared(self.real_lock_path)


    def launch_synt_workloads(self) -> Optional[subprocess.Popen]:
        raise NotImplementedError("Subclasses should implement this method.")

    def launch_real_workloads(self) -> subprocess.Popen:
        process = run_bash_commands(
            self.workload_arg,
            self.vm_arg,
            self.start_core_idx,
            self.data_dir,
            self.target_app,
            self.idx,
            self.real_lock_path,
        )
        return process

    def start(self):
        self.write_lock_start(self.synt_lock_path)
        self.write_lock_start(self.real_lock_path)

    def cleanup(self):
        cleanup_bash_commands(self.data_dir, self.target_app, self.idx)

    def colocate_real_with_synt(self):

        self.prepare_real_workloads()
        self.prepare_synt_workloads()

        real_process = self.launch_real_workloads()
        synt_process = self.launch_synt_workloads()

        self.wait_for_real_prepared()
        self.wait_for_synt_prepared()

        self.start()

        real_process.wait()
        if synt_process:
            synt_process.terminate()

        self.cleanup()


class IsolatedWorkloadLauncher(SyntheticWorkloadLauncher):
    def launch_synt_workloads(self) -> Optional[subprocess.Popen]:
        self.write_lock_start(self.synt_lock_path, value="1")
        return None

class StressEmulateWorkloadLauncher(SyntheticWorkloadLauncher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h5_path = kwargs.get("h5_path", "")
        self.fleetbench_path = kwargs.get("fleetbench_path", "./fleetbench")

        execution_file_path = os.path.join(self.fleetbench_path, "execution_plans")
        # Remove existing *.h5 files in the execution_file_path
        for file in os.listdir(execution_file_path):
            if file.endswith(".h5"):
                os.remove(os.path.join(execution_file_path, file))

        # Copy the h5_path to the execution_file_path
        if self.h5_path:
            subprocess.run(["cp", self.h5_path, execution_file_path], check=True)

    def launch_synt_workloads(self) -> Optional[subprocess.Popen]:
        # find which h5 file to launch.
        # lock file should be written by fleetbench code (sudo permission needed)
        # run stressors once the lock value has been changed
        process = subprocess.Popen(
            ["sudo", "bash", "-c", f"cd {self.fleetbench_path} && bash ./collect_stress_emulate_data.sh"],
            env={**os.environ, "START_PROFILING": "0"}
        )
        return process

web_list = webserver_list # + deathstartbench_list
bigdata_list = spark_list + database_list + kvstore_list
ml_list = ml_list
other_list = gapbs_list

MAX_CPU_CORES = 20
def write_configs_to_files(exp_configs, output_dir="exp_configs"):
    # Print the generated configurations
    for target_app, configs in exp_configs.items():
        os.makedirs(f"{output_dir}/{target_app}", exist_ok=True)
        with open(f"{output_dir}/{target_app}/config.txt", "w") as f:
            for config in configs:
                workload_arg = " ".join(config["workloads"])
                vm_arg = " ".join(map(str, config["vm_ids"]))
                f.write(f"\"{workload_arg}\" \"{vm_arg}\"\n")

def load_configs_from_files(config_dir):
    exp_configs = defaultdict(list)
    for num_vms in range(1, 7):
        config_path = f"{config_dir}/{num_vms}/config.txt"
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r") as f:
            for line in f:
                workload_part, vm_part = line.strip().split('" "')
                workloads = workload_part.strip('"').split()
                vm_ids = list(map(int, vm_part.strip('"').split()))
                exp_configs[num_vms].append({
                    "workloads": workloads,
                    "vm_ids": vm_ids,
                })
    return exp_configs

def get_workload_h5_path(base_path, exp_configs):
    for root, _, files in os.walk(base_path):
        files = sorted(files)
        for file in files:
            if file.endswith(".h5"):
                # parse_file_name plan_num_vms_6_6_node12.h5 to obtain (6, 6)
                prefix = "plan_num_vms_"
                suffix = "_node12.h5"
                start_idx = file.index(prefix) + len(prefix)
                end_idx = file.index(suffix)
                num_vms_str = file[start_idx:end_idx]
                num_vm, file_idx = tuple(map(int, num_vms_str.split("_")))
                print(num_vm, file_idx)
                exp_configs[num_vm][file_idx]["h5_path"] = os.path.join(root, file)

def get_num_assigned_cores(workloads, workload_scripts_dir="./workload_scripts"):
    assigned_cores = 0
    for workload in workloads:
        exp_config_path = os.path.join(workload_scripts_dir, workload, "exp_config.sh")
        with open(exp_config_path, "r") as f:
            for line in f:
                if "num_cores=" in line:
                    num_cores = int(line.strip().split("=")[1])
                    break
        assigned_cores += num_cores
    return assigned_cores

def run_bash_commands(workload_arg, vm_arg, start_core_idx, data_dir, target_app, exp_idx, lock_file_path):
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/{target_app}", exist_ok=True)
    os.makedirs(f"{data_dir}/{target_app}/{exp_idx}", exist_ok=True)

    # Clear the log path
    subprocess.run([f"sudo rm -rf {tacc_stats_log_path}/*"], shell=True, check=True)
    # Run the experiment
    process = subprocess.Popen(
        ["sudo", "bash", "./run_exps_noisy_neighbor_synt.sh", workload_arg, vm_arg, str(start_core_idx)],
        env={**os.environ, "LOCK_FILE_PATH": lock_file_path}
    )
    return process

def cleanup_bash_commands(data_dir, target_app, exp_idx):
    # Copy logs and results
    subprocess.run(["sudo", "cp", f"{tacc_stats_log_path}/current", f"{data_dir}/{target_app}/{exp_idx}/stats.txt"], check=True)
    subprocess.run(["sudo", "cp", f"{tacc_stats_log_path}/power_monitor.log", f"{data_dir}/{target_app}/{exp_idx}/power.log"], check=True)
    subprocess.run(f"sudo mv result_app_perf*.txt {data_dir}/{target_app}/{exp_idx}/", shell=True, check=True)

def get_workload_type(workload):
    if workload in webserver_list:
        return "webserver"
    elif workload in deathstartbench_list:
        return "deathstarbench"
    elif workload in spark_list:
        return "spark"
    elif workload in database_list:
        return "database"
    elif workload in kvstore_list:
        return "kvstore"
    elif workload in ml_list:
        return "ml"
    elif workload in gapbs_list:
        return "graph"
    else:
        return "unknown"


if __name__ == "__main__":
    tacc_stats_log_path = "/var/log/hpcperfstats"
    synt_types = ["isolated", "rl", "nn", "interpolation"]
    for synt_type in ["memcpy"]:
        target_dir = f"benchmarks_synt_{synt_type}"
        data_dir = f"data/{target_dir}"
        config_path = "./exp_configs/node12"
        workload_path = f"./synthetic_workloads/workload_mix/{synt_type}"
        fleetbench_path = "/users/dhkim/fleetbench"
        exp_configs = load_configs_from_files(config_path)

        get_workload_h5_path(workload_path, exp_configs)

        if synt_type == "isolated":
            SyntheticWorkloadLauncherClass = IsolatedWorkloadLauncher
        elif synt_type == "mimesys" or synt_type == "nn" or synt_type == "interpolation":
            SyntheticWorkloadLauncherClass = StressEmulateWorkloadLauncher

        new_exp_configs = defaultdict(list)

        for target_app in target_app_lists:
            target_workload_type = get_workload_type(target_app)
            for num_vms, configs in exp_configs.items():
                for exp_idx, config in enumerate(configs):
                    workloads = config["workloads"]
                    vm_ids = config["vm_ids"]
                    h5_path = config.get("h5_path", "")

                    candidate_vms = set(vms_per_workload[target_workload_type]) - set(vm_ids)
                    if not candidate_vms:
                        continue
                    vm_id = candidate_vms.pop()

                    num_assigned_cores = get_num_assigned_cores(workloads + [target_app])
                    if num_assigned_cores >= MAX_CPU_CORES:
                        print(f"Assigned cores {num_assigned_cores} exceed the maximum allowed {MAX_CPU_CORES}. Skipping this configuration.")
                        continue


                    new_workloads = [target_app]
                    new_vm_ids = [vm_id]
                    start_core_idx = get_num_assigned_cores(workloads)

                    new_exp_configs[target_app].append({
                        "real_workloads": workloads,
                        "workloads": new_workloads,
                        "vm_ids": new_vm_ids,
                        "start_core_idx": start_core_idx,
                        "h5_path": h5_path
                    })

        # write_configs_to_files(new_exp_configs, output_dir="exp_configs_noisy_neighbor_synt")

        workload_path = "./workload_scripts"
        shared_dir = "/dev/shm/shared"

        synt_lock_path = fleetbench_path + "/fleetbench/stress_emulate/memstrata_commands/synt_lock"
        # Print the generated configurations
        for target_app, workloads in new_exp_configs.items():
            for idx, workload in enumerate(reversed(workloads)):
                if idx >= 10:
                    continue
                adjusted_idx = len(workloads) - 1 - idx
                print(f"Target App: {target_app}, Workloads: {workload['workloads']}, VM IDs: {workload['vm_ids']}")
                workload_arg = " ".join([os.path.join(workload_path, wl) for wl in workload['workloads']])
                vm_arg = " ".join(map(str, workload['vm_ids']))
                start_core_idx = workload['start_core_idx']

                isolated_workload_launcher = SyntheticWorkloadLauncherClass(
                real_lock_path=f"/tmp/real_lock_{target_app}_{adjusted_idx}.lock",
                synt_lock_path=f"{synt_lock_path}_{target_app}_{adjusted_idx}.lock",
                workload_arg=workload_arg,
                vm_arg=vm_arg,
                start_core_idx=start_core_idx,
                data_dir=data_dir,
                target_app=target_app,
                idx=adjusted_idx,
                h5_path=workload.get("h5_path", ""),
                fleetbench_path=fleetbench_path
                )

                subprocess.run(["sudo", "cp", "create_symbolic_links.sh", shared_dir], check=True)
                isolated_workload_launcher.colocate_real_with_synt()
