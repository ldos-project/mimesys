#!/bin/bash
# Install Silo (TPC-C).  Source: yuhong-zhong/silo fork.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y \
    libjemalloc-dev libdb++-dev build-essential libaio-dev libnuma-dev \
    libssl-dev zlib1g-dev autoconf

cd "$BASE_PATH"
if [ ! -d "silo" ]; then
    git clone https://github.com/yuhong-zhong/silo.git
fi
cd silo
MODE=perf make -j dbtest
