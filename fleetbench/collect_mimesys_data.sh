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

for ((batch_idx=0; batch_idx<num_batches; batch_idx++)); do
    echo "Processing batch $((batch_idx + 1))/$num_batches..."

    # Clean the target directory at the start of each batch
    rm -f "${TARGET_DIR}/plan_"*.h5

    # Drop page cache and remove any stress-ng temp files from previous plan.
    # This ensures fallocate/sendfile start from a clean filesystem state each run.
    sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    rm -rf /tmp/stress-ng-* tmp-stress-ng-* 2>/dev/null || true

    # Flush NVMe/block-device write caches so no dirty data lingers in hardware.
    for dev in $(lsblk -dno NAME | grep -E '^(sd|nvme|vd)'); do
        sudo blockdev --flushbufs /dev/$dev 2>/dev/null || true
        # NVMe flush command (no-op on non-NVMe devices)
        sudo nvme flush /dev/$dev 2>/dev/null || true
    done

    # TRIM all mounted filesystems so the SSD GC can settle before the next run.
    # Without this, fallocate-triggered block deallocations pile up and GC fires
    # unpredictably mid-benchmark, inflating IO measurements.
    sudo fstrim -v / 2>/dev/null || true

    # Wait until block device IO quiesces (< 500 KB/s for 1 second).
    # Gives write-back from the previous plan time to drain fully.
    for dev in $(lsblk -dno NAME | grep -E '^(sd|nvme)'); do
        if [ ! -b /dev/$dev ]; then continue; fi
        while true; do
            before=$(awk -v d="$dev" '$3==d{print $6+$10}' /proc/diskstats 2>/dev/null || echo 0)
            sleep 1
            after=$(awk -v d="$dev" '$3==d{print $6+$10}' /proc/diskstats 2>/dev/null || echo 0)
            delta_kb=$(( (after - before) / 2 ))   # sectors -> KB
            [ "$delta_kb" -lt 500 ] && break
        done
    done

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
    MIMESYS_ITERS=${MIMESYS_ITERS:-1} MIMESYS_SLEEP=${MIMESYS_SLEEP:-1} ACTION_PROFILING_CACHE_DIR=${HOME_PATH}/fleetbench ACTION_LIST_PATH=${HOME_PATH}/fleetbench/fleetbench/mimesys/mimesys_actions.txt TACC_STATS_DIR=${HOME_PATH}/HPCPerfStats/monitor/src sudo bazel run --config=clang --config=opt fleetbench/mimesys:mimesys_benchmark -- --benchmark_filter="BM_Mimesys"
done

echo "Batch processing complete."
