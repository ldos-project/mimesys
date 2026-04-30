# Assume VMs are already created by "create_vm.sh" script

trap 'jobs -p | xargs -r kill' EXIT

workload="$1"
num_iters="$2"

echo "Running workload: $workload for $num_iters iterations"
workload_path="./workload_scripts/$workload"
tacc_stats_log_path="/var/log/hpcperfstats"

target_dir="benchmarks"
mkdir -p "data/$target_dir"

for ((i=1; i<num_iters; i++)); do
    echo "Iteration $i"
    sudo rm $tacc_stats_log_path/*
    sudo bash ./run_exp_numa.sh "$workload_path" 0

    mkdir -p data/$target_dir/$workload-$i
    sudo cp $tacc_stats_log_path/current data/$target_dir/$workload-$i/stats-$workload.txt
    sudo cp $tacc_stats_log_path/power_monitor.log data/$target_dir/$workload-$i/power-$workload.log
    sudo mv result_app_perf*.txt data/$target_dir/$workload-$i/
done
