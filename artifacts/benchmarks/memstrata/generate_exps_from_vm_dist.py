import random
from collections import defaultdict
import os
import subprocess

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

web_list = webserver_list # + deathstartbench_list
bigdata_list = spark_list + database_list + kvstore_list
ml_list = ml_list
other_list = gapbs_list

workload_proportion = {
    "web": 0.31,
    "bigdata": 0.32,
    "ml": 0.11,
    "other": 0.1,
}

vms_per_workload = {
    "webserver": [1, 2],
    "deathstarbench": [3],
    "spark": [4],
    "database": [6],
    "kvstore": [7, 8],
    "ml": [9],
    "graph": [10],
}

MAX_CPU_CORES = 20

def generate_experiment_configs():
    num_envs = 1
    exp_configs = defaultdict(list)

    for num_vms in range(1, 6):
        while len(exp_configs[num_vms]) < num_envs:
            workloads = []
            workload_types = []
            vm_ids = []

            for i in range(num_vms):
                workload_type = random.choices(
                    population=["web", "bigdata", "ml", "other"],
                    weights=[workload_proportion["web"], workload_proportion["bigdata"], workload_proportion["ml"], workload_proportion["other"]],
                    k=1,
                )[0]

                if workload_type == "web":
                    workload = random.choice(web_list)
                elif workload_type == "bigdata":
                    workload = random.choice(bigdata_list)
                elif workload_type == "ml":
                    workload = random.choice(ml_list)
                else:
                    workload = random.choice(other_list)


                workloads.append(workload)
                if workload in webserver_list:
                    workload_types.append("webserver")
                elif workload in deathstartbench_list:
                    workload_types.append("deathstarbench")
                elif workload in spark_list:
                    workload_types.append("spark")
                elif workload in database_list:
                    workload_types.append("database")
                elif workload in kvstore_list:
                    workload_types.append("kvstore")
                elif workload in ml_list:
                    workload_types.append("ml")
                elif workload in gapbs_list:
                    workload_types.append("graph")

            for workload_type in workload_types:
                candidate_vms = set(vms_per_workload[workload_type]) - set(vm_ids)
                if not candidate_vms:
                    continue
                vm_id = candidate_vms.pop()
                vm_ids.append(vm_id)

            if len(vm_ids) < num_vms:
                print("Not enough unique VM IDs available for the selected workloads. Skipping this configuration.")
                continue


            if any(set(config["workloads"]) == set(workloads) for config in exp_configs[num_vms]):
                print(f"Configuration with workloads {workloads} already exists. Skipping this configuration.")
                continue

            assigned_cores = 0
            for workload in workloads:
                exp_config_path = f"./workload_scripts/{workload}/exp_config.sh"
                with open(exp_config_path, "r") as f:
                    for line in f:
                        if "num_cores=" in line:
                            num_cores = int(line.strip().split("=")[1])
                            break
                assigned_cores += num_cores
            if assigned_cores > MAX_CPU_CORES:
                print(f"Assigned cores {assigned_cores} exceed the maximum allowed {MAX_CPU_CORES}. Skipping this configuration.")
                continue

            exp_configs[num_vms].append({
                "workloads": workloads,
                "vm_ids": vm_ids,
            })

    return exp_configs

def write_configs_to_files(exp_configs):
    output_dir = "exp_configs"
    # Print the generated configurations
    for num_vms, configs in exp_configs.items():
        os.makedirs(f"{output_dir}/{num_vms}", exist_ok=True)
        with open(f"{output_dir}/{num_vms}/config.txt", "w") as f:
            for config in configs:
                workload_arg = " ".join(config["workloads"])
                vm_arg = " ".join(map(str, config["vm_ids"]))
                f.write(f"\"{workload_arg}\" \"{vm_arg}\"\n")

def to_experiment_command(workloads, vm_ids):
    workload_path="./workload_scripts"
    workload_paths = [f"{workload_path}/{workload}" for workload in workloads]
    workload_arg = " ".join(workload_paths)
    vm_arg = " ".join(map(str, vm_ids))
    return f"sudo bash ./run_exps.sh \"{workload_arg}\" \"{vm_arg}\""

def run_bash_commands(workload_arg, vm_arg, num_vms, data_dir, exp_idx):
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/num_vms_{num_vms}", exist_ok=True)

    # Clear the log path
    subprocess.run([f"sudo rm -rf {tacc_stats_log_path}/*"], shell=True, check=True)
    # Run the experiment
    subprocess.run(["sudo", "bash", "./run_exps.sh", workload_arg, vm_arg], check=True)
    # Copy logs and results
    subprocess.run(["sudo", "cp", f"{tacc_stats_log_path}/current", f"{data_dir}/num_vms_{num_vms}/stats_{exp_idx}.txt"], check=True)
    subprocess.run(["sudo", "cp", f"{tacc_stats_log_path}/power_monitor.log", f"{data_dir}/num_vms_{num_vms}/power_{exp_idx}.log"], check=True)
    # subprocess.run(["sudo", "mv", "result_app_perf*.txt", f"{data_dir}/num_vms_{num_vms}/"], shell=True, check=True)

if __name__ == "__main__":
    exp_configs = generate_experiment_configs()
    write_configs_to_files(exp_configs)

    tacc_stats_log_path = "/var/log/hpcperfstats"
    target_dir = "benchmarks"
    data_dir = f"data/{target_dir}"

    for num_vms, configs in exp_configs.items():
        for config_idx, config in enumerate(configs):
            workloads = config["workloads"]
            vm_ids = config["vm_ids"]

            workload_path="./workload_scripts"
            workload_paths = [f"{workload_path}/{workload}" for workload in workloads]

            workload_arg = " ".join(workload_paths)
            vm_arg = " ".join(map(str, vm_ids))

            run_bash_commands(
                workload_arg=workload_arg,
                vm_arg=vm_arg,
                num_vms=num_vms,
                data_dir=data_dir,
                exp_idx=config_idx
            )
