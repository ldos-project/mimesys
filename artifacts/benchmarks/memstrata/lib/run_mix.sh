#!/bin/bash
# Unified mixed-workload runner.  Replaces:
#   - run_exp_numa.sh                  (--wait-policy all)
#   - run_exps_noisy_neighbor.sh       (--wait-policy target)
#   - run_exps_noisy_neighbor_synt.sh  (--wait-policy target --neighbor-mode synthetic)
#
# Usage:
#   run_mix.sh \
#       --workloads "silo_tpcc dacapo_tomcat" \
#       --vm-ids   "6 1" \
#       [--core-base 0] \
#       [--wait-policy all|target]   # default all
#       [--target-idx N]             # which workload (0-based) is "the target"; default last
#       [--neighbor-mode none|real|synthetic]   # default none
#       [--lock-file PATH]           # enables target<->neighbor sync handshake
#       [--sample-interval 1]
#       [--output-dir data/run-XX]
#
# The workload paths are resolved against ./workload_scripts/ unless an absolute
# path is given.

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT_DIR/config/host.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/ssh_helpers.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/vm_xml.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/samplers.sh"

######################################################
# Argument parsing
######################################################

WORKLOADS=""
VM_IDS=""
CORE_BASE=0
WAIT_POLICY="all"
TARGET_IDX=""
NEIGHBOR_MODE="none"
LOCK_FILE=""
SAMPLE_INTERVAL="${SAMPLE_INTERVAL_SEC:-1}"
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --workloads)        WORKLOADS="$2";       shift 2 ;;
        --vm-ids)           VM_IDS="$2";          shift 2 ;;
        --core-base)        CORE_BASE="$2";       shift 2 ;;
        --wait-policy)      WAIT_POLICY="$2";     shift 2 ;;
        --target-idx)       TARGET_IDX="$2";      shift 2 ;;
        --neighbor-mode)    NEIGHBOR_MODE="$2";   shift 2 ;;
        --lock-file)        LOCK_FILE="$2";       shift 2 ;;
        --sample-interval)  SAMPLE_INTERVAL="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";      shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

[ -n "$WORKLOADS" ] || { echo "--workloads is required" >&2; exit 2; }
[ -n "$VM_IDS" ]   || { echo "--vm-ids is required" >&2; exit 2; }

read -r -a workload_arr <<< "$WORKLOADS"
read -r -a vm_arr       <<< "$VM_IDS"

if [ "${#workload_arr[@]}" -ne "${#vm_arr[@]}" ]; then
    echo "--workloads and --vm-ids must have the same count" >&2
    exit 2
fi

# Resolve each workload to a workload_scripts/<name>/ path.
exp_arr=()
for w in "${workload_arr[@]}"; do
    if [[ "$w" = /* ]]; then
        exp_arr+=("$w")
    else
        exp_arr+=("$ROOT_DIR/workload_scripts/$w")
    fi
done

# Default target = last entry.
if [ -z "$TARGET_IDX" ]; then
    TARGET_IDX=$(( ${#exp_arr[@]} - 1 ))
fi

# Pick the neighbor module.
case "$NEIGHBOR_MODE" in
    none|real|synthetic)
        # shellcheck disable=SC1090
        source "$SCRIPT_DIR/neighbor/$NEIGHBOR_MODE.sh"
        ;;
    *) echo "Unknown --neighbor-mode: $NEIGHBOR_MODE" >&2; exit 2 ;;
esac

######################################################
# Clean target workload's previous results
######################################################

(cd "${exp_arr[0]}" && rm -f result_* || true)

######################################################
# Read per-workload VM sizing + compute CPU pinning
######################################################

num_vms="${#exp_arr[@]}"
num_vcpus_arr=()
vm_core_arr=()
vm_size_MB_arr=()

core_cursor="$CORE_BASE"
for exp in "${exp_arr[@]}"; do
    # shellcheck disable=SC1091
    source "$exp/exp_config.sh"
    num_vcpus_arr+=("$num_cores")
    vm_size_MB_arr+=("$vm_size_MB")
    vm_core_arr+=("$(vm_pin_cores "$num_cores" "$core_cursor")")
    core_cursor=$(( core_cursor + num_cores ))
done

######################################################
# Init host: sysctls + drop existing VMs
######################################################

echo 1800 | sudo tee /proc/sys/kernel/panic >/dev/null
echo      | sudo tee /sys/kernel/debug/tracing/trace >/dev/null
echo 0    | sudo tee /proc/sys/kernel/numa_balancing >/dev/null
echo 2    | sudo tee /sys/kernel/mm/ksm/run >/dev/null
echo 1    | sudo tee /proc/sys/vm/drop_caches >/dev/null

while [ -n "$(pgrep -fn 'collect-edp')" ]; do
    kill "$(pgrep -fn 'collect-edp')" || true
    sleep 1
done

for vm_idx in "${vm_arr[@]}"; do
    vm_kill "vm${vm_idx}"
done

bash "$ROOT_DIR/utils/disable_cpu_freq_scaling.sh"

######################################################
# Start each VM with the right pinning + memory
######################################################

for idx in "${!vm_arr[@]}"; do
    vm_idx="${vm_arr[$idx]}"
    vm_define_pinned \
        "vm${vm_idx}" \
        "${num_vcpus_arr[$idx]}" \
        "${vm_core_arr[$idx]}" \
        "${vm_size_MB_arr[$idx]}"
    sudo virsh start "vm${vm_idx}"
    vm_wait_ready "vm${vm_idx}"
done

######################################################
# Push prepare_exp.sh + run_exp.sh into each VM
######################################################

for idx in "${!vm_arr[@]}"; do
    vm_idx="${vm_arr[$idx]}"
    exp="${exp_arr[$idx]}"
    dom="vm${vm_idx}"

    vm_scp_to "$dom" "$exp/prepare_exp.sh" "/home/ubuntu/prepare_exp.sh"
    vm_scp_to "$dom" "$exp/run_exp.sh"     "/home/ubuntu/run_exp.sh"

    # util-linux 2.34 on focal mis-resolves the virtiofs tag; call mount(2) directly.
    vm_ssh "$dom" "mkdir -p /home/ubuntu/shared && if ! findmnt -t virtiofs /home/ubuntu/shared >/dev/null 2>&1; then sudo python3 -c \"import ctypes,os,sys; l=ctypes.CDLL('libc.so.6',use_errno=True); r=l.mount(b'shared',b'/home/ubuntu/shared',b'virtiofs',0,None); sys.exit(0 if r==0 else 'mount(2) failed: '+os.strerror(ctypes.get_errno()))\"; fi"
    vm_ssh "$dom" "bash /home/ubuntu/shared/create_symbolic_links.sh || true"
    vm_ssh "$dom" "source ~/.bashrc; bash /home/ubuntu/prepare_exp.sh"
done

######################################################
# Spawn neighbor (synthetic only; real/none are no-ops),
# then synchronise start via optional lock file.
######################################################

neighbor_prepare

if [ -n "$LOCK_FILE" ]; then
    sudo bash -c "echo 0 > '$LOCK_FILE'"
fi

samplers_reset_logs
samplers_start "$SAMPLE_INTERVAL"

# If we have a lock file, wait until neighbor side signals ready (writes "1")
# then write "2" to release everyone.  Without a lock file, just kick off.
if [ -n "$LOCK_FILE" ]; then
    neighbor_wait_ready "$LOCK_FILE"
    sudo bash -c "echo 2 > '$LOCK_FILE'"
fi

neighbor_start "$LOCK_FILE"

######################################################
# Launch every workload via SSH (background inside VM)
######################################################

for idx in "${!vm_arr[@]}"; do
    vm_idx="${vm_arr[$idx]}"
    dom="vm${vm_idx}"
    vm_ssh "$dom" "cd /home/ubuntu/; source ~/.bashrc; \
        GAPBS_DIR=/home/ubuntu/shared/gapbs \
        GAPBS_GRAPH_DIR=/home/ubuntu/shared/gapbs/benchmark/graphs \
        bash ./run_exp.sh > /home/ubuntu/result_run_exp_stdout.txt \
                          2> /home/ubuntu/result_run_exp_stderr.txt \
                          < /dev/null &"
done

######################################################
# Wait policy
######################################################

case "$WAIT_POLICY" in
    all)
        # Wait until every VM's run_exp has exited.
        for idx in "${!vm_arr[@]}"; do
            vm_idx="${vm_arr[$idx]}"
            dom="vm${vm_idx}"
            while [ -n "$(vm_ssh "$dom" 'pgrep -f run_exp')" ]; do
                sleep "$SAMPLE_INTERVAL"
            done
        done
        ;;
    target)
        # Wait for the designated target VM, then pkill the rest.
        target_vm="${vm_arr[$TARGET_IDX]}"
        target_dom="vm${target_vm}"
        while [ -n "$(vm_ssh "$target_dom" 'pgrep -f run_exp')" ]; do
            sleep "$SAMPLE_INTERVAL"
        done
        for idx in "${!vm_arr[@]}"; do
            if [ "$idx" -eq "$TARGET_IDX" ]; then continue; fi
            vm_idx="${vm_arr[$idx]}"
            vm_ssh "vm${vm_idx}" "pkill -f run_exp || true" || true
        done
        ;;
    *)
        echo "Unknown --wait-policy: $WAIT_POLICY" >&2; exit 2 ;;
esac

samplers_stop
neighbor_cleanup

######################################################
# Collect per-VM result_app_perf.txt + sampler logs
######################################################

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$ROOT_DIR/$DATA_DIR/run-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

for idx in "${!vm_arr[@]}"; do
    vm_idx="${vm_arr[$idx]}"
    exp="${exp_arr[$idx]}"
    exp_name="$(basename "$exp")"
    vm_scp_from "vm${vm_idx}" \
        "/home/ubuntu/result_app_perf.txt" \
        "$OUTPUT_DIR/result_app_perf_${vm_idx}_${exp_name}.txt" || true
done

samplers_archive "$OUTPUT_DIR"

######################################################
# Cleanup VMs
######################################################

for vm_idx in "${vm_arr[@]}"; do
    vm_kill "vm${vm_idx}"
done

echo "Results in: $OUTPUT_DIR"
