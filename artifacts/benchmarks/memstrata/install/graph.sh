#!/bin/bash
# Install GAPBS into the virtiofs share so every VM can read the same graphs.
# Graph data files (road / urand / web) still need to be generated/downloaded
# per upstream GAPBS instructions.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y build-essential

cd "$BASE_PATH"
if [ ! -d "gapbs" ]; then
    git clone https://github.com/sbeamer/gapbs.git
fi
cd gapbs && make
