#!/bin/bash
# DLRM (Pond-style recommendation workload).  Requires Pond artifacts which
# are not redistributable here -- this is a stub that documents the manual
# steps.  Run inside the VM.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

cat <<'EOF'
[install/dlrm.sh]
The DLRM workloads (dlrm, dlrm_rm1_high/med/low, dlrm_rm2_1_*) depend on Pond's
pre-built dlrm artifacts and a conda environment named 'dlrm_cpu'.  Follow the
Pond instructions to populate $HOME/dlrm and $HOME/dlrm/paths.export, then
re-run each workload's prepare_exp.sh.  This installer only ensures the Pond
checkout exists.
EOF
