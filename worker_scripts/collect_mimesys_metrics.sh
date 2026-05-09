#!/bin/bash
# Run a chunk's plans via collect_mimesys_data.sh, zip the per-plan stats files,
# and ship the zip back to the controller. For robustness:
#   * any failure aborts the script (set -e + pipefail) — never silently
#     proceed past a broken zip or scp,
#   * the local zip and results/ directory are only deleted *after* the scp
#     to the controller succeeds; if the scp fails after retries, both are
#     left in place so the user can recover them manually.

set -euo pipefail

username="$1"
remote_host="$2"
remote_destination="$3"
chunk_idx="$4"
host_name=$(hostname)

fleetbench_dir="/users/$username/fleetbench"
cd "$fleetbench_dir"
bash collect_mimesys_data.sh

cd "$HOME"
target_dir="validation-$chunk_idx"
zip -r "$target_dir.zip" results

# Retry the scp because transient SSH/network errors during long collection runs are common.
attempt=0
max_attempts=5
until scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=30 \
        "$target_dir.zip" \
        "$username@$remote_host:$remote_destination/$host_name-$target_dir.zip"; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "scp failed after $max_attempts attempts; preserving $HOME/$target_dir.zip and $HOME/results/ for manual recovery." >&2
        exit 1
    fi
    sleep_seconds=$((attempt * 30))
    echo "scp attempt $attempt failed; retrying in ${sleep_seconds}s..." >&2
    sleep $sleep_seconds
done

sudo rm -rf results/*
rm -f "$target_dir.zip"
