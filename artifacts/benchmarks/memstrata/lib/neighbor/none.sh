#!/bin/bash
# Neighbor source: none / isolated.
#
# In this mode there is no co-running synthetic workload.  The only thing
# neighbor_start() does is flip the lock to "1" so the (optional) sync code
# in run_mix.sh proceeds.  Used when --neighbor-mode is omitted entirely or
# set to "isolated".

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
