#!/bin/bash
# pg-tpch (yuhong-zhong/pg-tpch): builds a custom PostgreSQL 9.3 + populates
# the TPC-H schema in perfdata-10GB/.  ./tpch_prepare takes a long time the
# first run (compiles Postgres + generates ~10 GB of TPC-H data); subsequent
# runs are no-ops.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

# Build deps pulled from pg-tpch's README.
sudo apt-get install -y \
    build-essential bison flex \
    libreadline-dev zlib1g-dev libxslt1-dev

cd "$BASE_PATH"
if [ ! -d "pg-tpch" ]; then
    git clone https://github.com/yuhong-zhong/pg-tpch.git
fi
cd pg-tpch
./tpch_prepare
