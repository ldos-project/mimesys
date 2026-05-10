#!/bin/bash
# Neighbor source: synthetic workloads driven by fleetbench's
# collect_stress_emulate_data.sh, replacing the real co-running mix.
#
# Required env vars:
#   FLEETBENCH_PATH  path to the fleetbench checkout
#   SYNT_H5_PATH     path to the .h5 execution plan to install (optional)
#   SYNT_LOCK_PATH   lock file fleetbench polls / writes (required)
#
# Protocol:
#   1. neighbor_prepare(): clear stale .h5 plans, copy SYNT_H5_PATH in,
#                          initialize SYNT_LOCK_PATH to 0.
#   2. fleetbench is launched as a child process (NEIGHBOR_PID returned).
#   3. fleetbench bumps SYNT_LOCK_PATH from 0 -> 1 when it's ready.
#   4. run_mix.sh writes "2" to SYNT_LOCK_PATH to release fleetbench's
#      stressors at the same instant it releases the real workload.
#
# This matches the contract of the old StressEmulateWorkloadLauncher.

NEIGHBOR_PID=""

neighbor_prepare() {
    : "${FLEETBENCH_PATH:?FLEETBENCH_PATH must be set for synthetic neighbor}"
    : "${SYNT_LOCK_PATH:?SYNT_LOCK_PATH must be set for synthetic neighbor}"

    local plan_dir="$FLEETBENCH_PATH/execution_plans"
    mkdir -p "$plan_dir"
    find "$plan_dir" -maxdepth 1 -name '*.h5' -delete

    if [ -n "${SYNT_H5_PATH:-}" ] && [ -f "$SYNT_H5_PATH" ]; then
        cp "$SYNT_H5_PATH" "$plan_dir/"
    fi

    local lock_dir
    lock_dir="$(dirname "$SYNT_LOCK_PATH")"
    sudo find "$lock_dir" -maxdepth 1 -name '*.lock' -delete 2>/dev/null || true
    sudo bash -c "echo 0 > '$SYNT_LOCK_PATH'"

    NEIGHBOR_PID=""
    (
        cd "$FLEETBENCH_PATH" && \
        START_PROFILING=0 sudo -E bash ./collect_stress_emulate_data.sh
    ) &
    NEIGHBOR_PID=$!
}

neighbor_wait_ready() {
    local lock_file="${1:-$SYNT_LOCK_PATH}"
    while [ "$(cat "$lock_file" 2>/dev/null)" != "1" ]; do
        sleep 1
    done
}

neighbor_start() {
    local lock_file="${1:-$SYNT_LOCK_PATH}"
    sudo bash -c "echo 2 > '$lock_file'"
}

neighbor_cleanup() {
    if [ -n "$NEIGHBOR_PID" ] && kill -0 "$NEIGHBOR_PID" 2>/dev/null; then
        kill "$NEIGHBOR_PID" 2>/dev/null || true
        wait "$NEIGHBOR_PID" 2>/dev/null || true
    fi
    NEIGHBOR_PID=""
}
