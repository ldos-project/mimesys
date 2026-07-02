#!/bin/bash
trap 'jobs -p | xargs -r kill' EXIT

set -eux -o pipefail

# Configurable variables
SOURCE_DIR="execution_plans"
TARGET_DIR="fleetbench/mimesys/execution_plans"
RUNFILES_DIR="/users/${USER:-dhkim}/.cache/bazel/_bazel_root/4ab79bdb1c421100af776bebe721ba14/execroot/_main/bazel-out/k8-opt-clang/bin/fleetbench/mimesys/mimesys_benchmark.runfiles/_main/fleetbench/mimesys/execution_plans"
BATCH_SIZE=1

files=($(ls ${SOURCE_DIR}/plan_[0-9][0-9][0-9][0-9][0-9][0-9].h5 | sort))
total_files=${#files[@]}
num_batches=$(( (total_files + BATCH_SIZE - 1) / BATCH_SIZE ))

MIMESYS_MIN_PLANS=${MIMESYS_MIN_PLANS:-1}
if [ "$total_files" -lt "$MIMESYS_MIN_PLANS" ]; then
    echo "ABORT: only $total_files plans found in $SOURCE_DIR (min=$MIMESYS_MIN_PLANS)" >&2
    exit 11
fi
echo "collect_mimesys_data: processing $total_files plans"

for ((batch_idx=0; batch_idx<num_batches; batch_idx++)); do
    echo "Processing batch $((batch_idx + 1))/$num_batches..."

    # Clean BOTH the target source dir AND the bazel runfiles execution_plans dir.
    # Without cleaning runfiles, stale plan_XXXXX.h5 symlinks left over from
    # previous invocations get processed by the binary alongside/instead of the
    # newly-copied plan — RL rewards then correspond to whatever was staged into
    # runfiles at first bazel-run time, not the current plan.
    sudo rm -f "${TARGET_DIR}/plan_"*.h5
    sudo rm -f "${RUNFILES_DIR}/plan_"*.h5

    rm -rf /tmp/stress-ng-* tmp-stress-ng-* 2>/dev/null || true

    start_idx=$((batch_idx * BATCH_SIZE))
    end_idx=$((start_idx + BATCH_SIZE))
    if [ $end_idx -gt $total_files ]; then
        end_idx=$total_files
    fi

    for ((i=start_idx; i<end_idx; i++)); do
        sudo cp "${files[i]}" "${TARGET_DIR}/"
        # Also stage the plan into bazel runfiles so the binary's Runfiles::Rlocation
        # sees it. bazel's own runfile refresh isn't reliable across successive
        # sudo bazel run invocations when the plan filename changes.
        sudo ln -sf "$(readlink -f "${TARGET_DIR}/$(basename ${files[i]})")" \
            "${RUNFILES_DIR}/$(basename ${files[i]})"
    done

    echo "Running commands on batch $((batch_idx + 1))..."
    HOME_PATH=${HOME_PATH:-$HOME}
    MIMESYS_ITERS=${MIMESYS_ITERS:-1} MIMESYS_SLEEP=${MIMESYS_SLEEP:-1} ACTION_PROFILING_CACHE_DIR=${HOME_PATH}/fleetbench ACTION_LIST_PATH=${ACTION_LIST_PATH:-${HOME_PATH}/fleetbench/fleetbench/mimesys/mimesys_actions.txt} TACC_STATS_DIR=${HOME_PATH}/HPCPerfStats/monitor/src sudo bazel run --config=clang --config=opt fleetbench/mimesys:mimesys_benchmark -- --benchmark_filter="BM_Mimesys" --benchmark_min_time=1.8s
done

echo "Batch processing complete."
