#!/bin/bash
trap 'jobs -p | xargs -r kill' EXIT

set -eux -o pipefail

# Configurable variables
SOURCE_DIR="execution_plans"
TARGET_DIR="fleetbench/mimesys/execution_plans"
# BATCH_SIZE=1: run one plan per benchmark invocation so filesystem state
# (fallocate temp files, page cache) is fully reset between plans.
BATCH_SIZE=1

# Find all matching files and sort them
files=($(ls ${SOURCE_DIR}/plan_[0-9][0-9][0-9][0-9][0-9][0-9].h5 | sort))
total_files=${#files[@]}
num_batches=$(( (total_files + BATCH_SIZE - 1) / BATCH_SIZE ))

# Guard against truly empty unzips (the original race had total_files=1 with stale
# stats files in ~/results being shipped back). Now that collect_mimesys_metrics.sh
# wipes ~/results at start, a low plan count just means a small RL chunk — only abort
# at zero. Caller can raise MIMESYS_MIN_PLANS for collection-only stricter checks.
MIMESYS_MIN_PLANS=${MIMESYS_MIN_PLANS:-1}
if [ "$total_files" -lt "$MIMESYS_MIN_PLANS" ]; then
    echo "ABORT: only $total_files plans found in $SOURCE_DIR (min=$MIMESYS_MIN_PLANS)" >&2
    exit 11
fi
echo "collect_mimesys_data: processing $total_files plans"

for ((batch_idx=0; batch_idx<num_batches; batch_idx++)); do
    echo "Processing batch $((batch_idx + 1))/$num_batches..."

    # Clean the target directory at the start of each batch
    rm -f "${TARGET_DIR}/plan_"*.h5

    # Remove any stress-ng temp files leaked by a previous plan.
    # (Cheap; no measurable IO impact.)
    # Pre-plan drop_caches / blockdev-flush / nvme-flush / fstrim / IO-quiesce-wait
    # was removed after measurement showed it INFLATED low-level IO variance ~10x
    # (fstrim TRIMs and journal commits get counted in /proc/diskstats during the
    # next measurement window). Real-IO workloads (Hdd_1MB ~290k bytes/window) were
    # unaffected; non-IO workloads' "phantom IO" dropped from ~300 → ~200 baseline.
    rm -rf /tmp/stress-ng-* tmp-stress-ng-* 2>/dev/null || true

    # Calculate range for this batch
    start_idx=$((batch_idx * BATCH_SIZE))
    end_idx=$((start_idx + BATCH_SIZE))
    if [ $end_idx -gt $total_files ]; then
        end_idx=$total_files
    fi

    # Copy files for current batch
    for ((i=start_idx; i<end_idx; i++)); do
        cp "${files[i]}" "${TARGET_DIR}/"
    done

    # Run your commands for each batch here
    echo "Running commands on batch $((batch_idx + 1))..."
    HOME_PATH=${HOME_PATH:-$HOME}
    MIMESYS_ITERS=${MIMESYS_ITERS:-1} MIMESYS_SLEEP=${MIMESYS_SLEEP:-1} ACTION_PROFILING_CACHE_DIR=${HOME_PATH}/fleetbench ACTION_LIST_PATH=${HOME_PATH}/fleetbench/fleetbench/mimesys/mimesys_actions.txt TACC_STATS_DIR=${HOME_PATH}/HPCPerfStats/monitor/src sudo bazel run --config=clang --config=opt fleetbench/mimesys:mimesys_benchmark -- --benchmark_filter="BM_Mimesys" --benchmark_min_time=1.8s
done

echo "Batch processing complete."
