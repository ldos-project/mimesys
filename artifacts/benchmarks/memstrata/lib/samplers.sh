#!/bin/bash
# Start / stop the per-second samplers: hpcperfstatsd (tacc_stats) and a
# powercap energy reader.  Both append to $TACC_STATS_LOG_PATH.
# Interval is controlled by $SAMPLE_INTERVAL_SEC (default 1).

# Globals set by samplers_start so samplers_stop can kill the right PIDs.
SAMPLER_TACC_PID=""
SAMPLER_POWER_PID=""

samplers_start() {
    local interval="${1:-${SAMPLE_INTERVAL_SEC:-1}}"

    sudo "$TACC_STATS_DIR/hpcperfstatsd" begin

    (
        while true; do
            sudo "$TACC_STATS_DIR/hpcperfstatsd" collect
            sleep "$interval"
        done
    ) &
    SAMPLER_TACC_PID=$!

    sudo bash -c '
        LOGFILE="'"$TACC_STATS_LOG_PATH"'/power_monitor.log"
        INTERVAL="'"$interval"'"
        while true; do
            tstamp=$(date +%s.%6N)
            total_uj=0
            for f in /sys/class/powercap/*/energy_uj; do
                if [[ -r "$f" ]]; then
                    val=$(cat "$f")
                    total_uj=$((total_uj + val))
                fi
            done
            total_j=$(awk "BEGIN {printf \"%.6f\", $total_uj/1000000}")
            echo "$tstamp,$total_j" >> "$LOGFILE"
            sleep "$INTERVAL"
        done
    ' &
    SAMPLER_POWER_PID=$!
}

samplers_stop() {
    if [ -n "$SAMPLER_TACC_PID" ]; then
        sudo kill "$SAMPLER_TACC_PID" 2>/dev/null || true
        SAMPLER_TACC_PID=""
    fi
    if [ -n "$SAMPLER_POWER_PID" ]; then
        sudo kill "$SAMPLER_POWER_PID" 2>/dev/null || true
        SAMPLER_POWER_PID=""
    fi
}

# Wipe the tacc_stats log dir before a run.
samplers_reset_logs() {
    sudo rm -rf "$TACC_STATS_LOG_PATH"/* 2>/dev/null || true
}

# Copy current stats + power log into a result directory.
samplers_archive() {
    local outdir="$1"
    mkdir -p "$outdir"
    sudo cp "$TACC_STATS_LOG_PATH/current" "$outdir/stats.txt" 2>/dev/null || true
    sudo cp "$TACC_STATS_LOG_PATH/power_monitor.log" "$outdir/power.log" 2>/dev/null || true
}
