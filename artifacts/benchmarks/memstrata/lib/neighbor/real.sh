#!/bin/bash
# Neighbor source: real workloads running in real VMs.
#
# All neighbor VMs are already started by run_mix.sh -- this module is a
# no-op except for satisfying the optional lock-file handshake protocol.
# Use this when you want to measure a target app against actual co-running
# workloads from workload_scripts/.

neighbor_prepare() { :; }

neighbor_start() {
    local lock_file="$1"
    if [ -n "$lock_file" ]; then
        sudo bash -c "echo 1 > '$lock_file'"
    fi
}

neighbor_wait_ready() {
    local lock_file="$1"
    if [ -z "$lock_file" ]; then return 0; fi
    while [ "$(cat "$lock_file" 2>/dev/null)" != "1" ]; do
        sleep 1
    done
}

neighbor_cleanup() { :; }
