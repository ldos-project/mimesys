#!/bin/bash
# Iterate over every mix line in exp_configs/<config_dir>/<N>/config.txt and
# invoke `mimebench.py run-mix` on each, dumping results under
# data/<out>/N_<n>_idx_<i>/.
#
# Skips mixes whose output dir already has stats.txt + power.log (so the script
# is resumable after interruptions). Caps each mix's wall time via `timeout`
# so a stuck VM doesn't block the whole sweep.

set -u -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG_DIR="${CFG_DIR:-exp_configs/generated}"
OUT_BASE="${OUT_BASE:-data/profile-all}"
INTERVAL="${INTERVAL:-1}"
PER_MIX_TIMEOUT="${PER_MIX_TIMEOUT:-1800}"   # 30 minutes / mix
LOG_DIR="${LOG_DIR:-/tmp/mimebench_logs}"

mkdir -p "$LOG_DIR" "$OUT_BASE"

t_overall_start=$(date +%s)
total_mixes=0
skipped=0
ran=0
failed=0

for cfg in "$CFG_DIR"/*/config.txt; do
    n=$(basename "$(dirname "$cfg")")
    idx=0
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        wl=$(printf '%s' "$line" | awk -F'"' '{print $2}')
        vm=$(printf '%s' "$line" | awk -F'"' '{print $4}')
        out="$OUT_BASE/N_${n}/mix_${idx}"
        total_mixes=$((total_mixes + 1))

        if [ -f "$out/stats.txt" ] && [ -f "$out/power.log" ]; then
            printf "[N=%s mix=%d] SKIP (already done): %s\n" "$n" "$idx" "$wl"
            skipped=$((skipped + 1))
            idx=$((idx + 1))
            continue
        fi

        printf "\n[N=%s mix=%d] workloads='%s' vm-ids='%s'\n" "$n" "$idx" "$wl" "$vm"
        mkdir -p "$out"
        log="$LOG_DIR/run-mix-N${n}-mix${idx}.log"
        t0=$(date +%s)

        if timeout "$PER_MIX_TIMEOUT" ./mimebench.py run-mix \
                --workloads $wl \
                --vm-ids   $vm \
                --interval "$INTERVAL" \
                --out "$out" \
                </dev/null >"$log" 2>&1; then
            t1=$(date +%s); dt=$((t1 - t0))
            printf "[N=%s mix=%d] OK  (%ds, log=%s)\n" "$n" "$idx" "$dt" "$log"
            ran=$((ran + 1))
        else
            rc=$?
            t1=$(date +%s); dt=$((t1 - t0))
            printf "[N=%s mix=%d] FAIL rc=%d (%ds, log=%s)\n" "$n" "$idx" "$rc" "$dt" "$log"
            failed=$((failed + 1))
            # Make sure no VMs are left running between mixes.
            for vmid in $vm; do sudo virsh destroy "vm$vmid" >/dev/null 2>&1 || true; done
        fi
        idx=$((idx + 1))
    done < "$cfg"
done

t_overall=$((($(date +%s) - t_overall_start)))
printf "\n=== SWEEP DONE ===\n"
printf "total=%d ran=%d skipped=%d failed=%d wall=%ds\n" \
    "$total_mixes" "$ran" "$skipped" "$failed" "$t_overall"
