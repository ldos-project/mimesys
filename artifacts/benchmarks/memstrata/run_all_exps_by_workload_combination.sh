# Assume VMs are already created by "create_vm.sh" script

trap 'jobs -p | xargs -r kill' EXIT

num_iters="$1"

spec_workload_list=(
602.gcc_s
603.bwaves_s
605.mcf_s
607.cactuBSSN_s
619.lbm_s
631.deepsjeng_s
638.imagick_s
649.fotonik3d_s
654.roms_s
657.xz_s
)

gap_workload_list=(
bc-urand
bc-web
bfs-urand
bfs-web
cc-urand
cc-web
pr-urand
pr-web
)

workload_list=$"
dacapo_cassandra
dacapo_kafka
dacapo_spring
dacapo_tomcat
deathstarbench_media
deathstarbench_social
faster_uniform_ycsb_a
faster_uniform_ycsb_b
faster_uniform_ycsb_c
faster_uniform_ycsb_f
faster_ycsb_a
faster_ycsb_b
faster_ycsb_c
faster_ycsb_f
finagle_chirper
finagle_http
memcached_uniform_ycsb_a
memcached_ycsb_a
memcached_ycsb_b
memcached_ycsb_c
memcached_ycsb_d
memcached_ycsb_e
memcached_ycsb_f
redis_uniform_ycsb_a
redis_ycsb_a
redis_ycsb_b
redis_ycsb_c
redis_ycsb_d
redis_ycsb_e
redis_ycsb_f
silo_tpcc
spark_als
spark_gbt
spark_lda
spark_lr
spark_pca
spark_rf
spark_sort
spark_svd
spark_terasort
spark_wordcount
"

tpc_workload_list=$"
tpch_1
tpch_10
tpch_11
tpch_12
tpch_13
tpch_14
tpch_15
tpch_16
tpch_17
tpch_18
tpch_19
tpch_2
tpch_20
tpch_21
tpch_22
tpch_3
tpch_4
tpch_5
tpch_6
tpch_7
tpch_8
tpch_9
tpch_m
"

deathstarbench_list=$"
deathstarbench_media
deathstarbench_social
"

spark_list=$"
spark_als
spark_gbt
spark_lda
spark_lr
spark_pca
spark_rf
spark_sort
spark_svd
spark_terasort
spark_wordcount
"

memcached_list=$"
memcached_uniform_ycsb_a
memcached_ycsb_a
memcached_ycsb_b
memcached_ycsb_c
memcached_ycsb_d
memcached_ycsb_e
memcached_ycsb_f
"

tacc_stats_dir="/users/dhkim/HPCPerfStats/monitor/src"
tacc_stats_log_path="/var/log/hpcperfstats"


multi_list_combinations() {
  local arrays=("$@")  # Receive multiple arrays as arguments (each multiline)

  _combine() {
    local prefix="$1"
    shift
    if [ $# -eq 0 ]; then
      echo "$prefix"
      return
    fi

    local current_list="$1"
    shift

    # Read current list into array
    IFS=$'\n' read -r -d '' -a items <<< "$current_list"$'\0'

    for item in "${items[@]}"; do
      if [ -z "$prefix" ]; then
        _combine "$item" "$@"
      else
        _combine "$prefix $item" "$@"
      fi
    done
  }

  _combine "" "${arrays[@]}"
}

mapfile -t combinations_list < <(multi_list_combinations "$spark_list" "$deathstarbench_list")

for workload in "${combinations_list[@]}"; do
    echo "Running workload combination: $workload"

    workload_paths=()
    for w in $workload; do
        workload_paths+=(./workload_scripts/$w)
    done

    target_dir="benchmarks"
    mkdir -p "data/$target_dir"

    for ((i=1; i<num_iters; i++)); do
        echo "Iteration $i"
        sudo rm $tacc_stats_log_path/*

        sudo bash ./run_exp_numa.sh "${workload_paths[*]}" 0

        workload_joined=$(echo "$workload" | tr ' ' '_')
        mkdir -p data/$target_dir/$workload_joined-$i
        sudo cp $tacc_stats_log_path/current data/$target_dir/$workload_joined-$i/stats-$workload_joined.txt
        sudo cp $tacc_stats_log_path/power_monitor.log data/$target_dir/$workload_joined-$i/power-$workload_joined.log
        sudo mv result_app_perf*.txt data/$target_dir/$workload_joined-$i/
    done
done
