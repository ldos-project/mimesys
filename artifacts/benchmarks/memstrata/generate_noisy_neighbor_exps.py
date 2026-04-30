import random
from collections import defaultdict
import os
import subprocess
from copy import deepcopy

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

def run_bash_commands(workload_arg, vm_arg, data_dir, target_app, exp_idx):
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/{target_app}", exist_ok=True)
    os.makedirs(f"{data_dir}/{target_app}/{exp_idx}", exist_ok=True)

    # Clear the log path
    subprocess.run([f"sudo rm -rf {tacc_stats_log_path}/*"], shell=True, check=True)
    # Run the experiment
    subprocess.run(["sudo", "bash", "./run_exps_noisy_neighbor.sh", workload_arg, vm_arg], check=True)
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
    target_dir = "benchmarks"
    data_dir = f"data/{target_dir}"
    shared_dir = "/dev/shm/shared"

    config_path = "./exp_configs/node12"
    exp_configs = load_configs_from_files(config_path)

    new_exp_configs = defaultdict(list)

    for target_app in target_app_lists:
        target_workload_type = get_workload_type(target_app)
        for num_vms, configs in exp_configs.items():
            for exp_idx, config in enumerate(configs):
                workloads = config["workloads"]
                vm_ids = config["vm_ids"]

                candidate_vms = set(vms_per_workload[target_workload_type]) - set(vm_ids)
                if not candidate_vms:
                    continue
                vm_id = candidate_vms.pop()

                print(workloads)
                new_workloads = deepcopy(workloads)
                new_workloads.append(target_app)

                new_vm_ids = deepcopy(vm_ids)
                new_vm_ids.append(vm_id)
                print(target_app, new_workloads)

                num_assigned_cores = get_num_assigned_cores(new_workloads)
                if num_assigned_cores > MAX_CPU_CORES:
                    print(f"Assigned cores {num_assigned_cores} exceed the maximum allowed {MAX_CPU_CORES}. Skipping this configuration.")
                    continue

                new_exp_configs[target_app].append({
                    "workloads": new_workloads,
                    "vm_ids": new_vm_ids,
                })

    write_configs_to_files(new_exp_configs, output_dir="exp_configs_noisy_neighbor")

    workload_path = "./workload_scripts"

     # Print the generated configurations
    for target_app, workloads in new_exp_configs.items():
        for idx, workload in enumerate(reversed(workloads)):
            if idx >= 10:
                continue
            adjusted_idx = len(workloads) - 1 - idx
            print(f"Target App: {target_app}, Workloads: {workload['workloads']}, VM IDs: {workload['vm_ids']}")
            workload_arg = " ".join([os.path.join(workload_path, wl) for wl in workload['workloads']])
            vm_arg = " ".join(map(str, workload['vm_ids']))

            subprocess.run(["sudo", "cp", "create_symbolic_links.sh", shared_dir], check=True)
            run_bash_commands(workload_arg, vm_arg, data_dir, target_app, adjusted_idx)
