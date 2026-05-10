#!/bin/bash
# Dispatcher: ./install.sh STACK [STACK ...]
# Each STACK matches one of the files alongside this script.
# Run *inside* the VM, as user ubuntu, after the virtiofs share is mounted at
# /home/ubuntu/shared.

set -eu -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 STACK [STACK ...]" >&2
    echo "Available stacks:" >&2
    for f in "$SCRIPT_DIR"/*.sh; do
        b="$(basename "$f" .sh)"
        case "$b" in install|_common) continue ;; esac
        echo "  - $b" >&2
    done
    exit 2
fi

for stack in "$@"; do
    script="$SCRIPT_DIR/$stack.sh"
    if [ ! -f "$script" ]; then
        echo "Unknown stack: $stack" >&2
        exit 2
    fi
    echo "=== installing: $stack ==="
    bash "$script"
done
