#!/bin/bash
# SPEC CPU2017 is non-redistributable; install must be done by hand.
# This stub fails loudly so the operator notices before a run.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

cat <<'EOF'
[install/spec.sh]
SPEC CPU2017 cannot be downloaded automatically.  Follow the Pond
instructions (https://github.com/MoatLab/Pond/tree/master/cpu2017) to install
the suite inside this VM, then re-run any workload's prepare_exp.sh to verify.
EOF
exit 0
