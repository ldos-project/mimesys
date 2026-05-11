#!/bin/bash
# Install GAPBS into the virtiofs share so every VM can read the same graphs.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y build-essential

cd "$BASE_PATH"
if [ ! -d "gapbs" ]; then
    git clone https://github.com/sbeamer/gapbs.git
fi
cd gapbs
make

# Restrict bench.mk to the road graph only (we don't use twitter/web/kron/urand
# and downloading them is slow + bandwidth-heavy). Idempotent: re-running the
# sed against an already-narrowed file is a no-op.
sed -i 's|^GRAPHS = .*|GRAPHS = road|' benchmark/bench.mk
make bench-graphs
