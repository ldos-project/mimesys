#!/bin/bash
# Linking every top-level dir in /home/ubuntu/shared/ as ~/<name> so workload
# scripts can use plain relative paths (e.g. `cd YCSB`). Idempotent via -f.
for dir in /home/ubuntu/shared/*; do
    name="$(basename "$dir")"
    [ "$name" = "create_symbolic_links.sh" ] && continue
    ln -sfn "$dir" "$name"
done
