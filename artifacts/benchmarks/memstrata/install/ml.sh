#!/bin/bash
# Install MLPerf inference pieces (sdxl, resnet50).
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y python3-pip python3-venv llvm libgl1-mesa-glx
pip install mlc-scripts networkx fsspec sympy

"$HOME/.local/bin/mlcr" install,python-venv --name=mlperf
export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"

yes "" | "$HOME/.local/bin/mlcr" \
    run-mlperf,inference,_find-performance,_full,_r5.1-dev \
    --model=sdxl \
    --implementation=reference \
    --framework=pytorch \
    --category=datacenter \
    --scenario=Offline \
    --device=cpu \
    --actions=download,preprocess

yes "" | "$HOME/.local/bin/mlcr" \
    run-mlperf,inference,_find-performance,_full,_r5.1-dev \
    --model=resnet50 \
    --implementation=reference \
    --framework=onnxruntime \
    --category=datacenter \
    --scenario=Offline \
    --execution_mode=test \
    --device=cpu \
    --actions=download,preprocess
