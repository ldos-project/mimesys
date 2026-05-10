#!/bin/bash
# PostgreSQL + pg-tpch.  Stub: the workload scripts assume an existing
# /home/ubuntu/pg-tpch checkout populated with the chosen scale-factor data.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y postgresql postgresql-contrib build-essential git
cat <<'EOF'
[install/tpch.sh]
TPC-H query workloads (tpch_*) expect /home/ubuntu/pg-tpch to be populated
with the runner scripts and pre-generated data.  Mirror the layout from the
upstream Pond/pg-tpch instructions, then re-run any workload's prepare_exp.sh
to verify connectivity.
EOF
