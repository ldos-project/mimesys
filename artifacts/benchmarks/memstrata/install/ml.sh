#!/bin/bash
# Install MLPerf inference pieces (sdxl, resnet50).
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y python3-pip python3-venv llvm libgl1-mesa-glx
pip install mlc-scripts networkx fsspec sympy

# Put the MLC cache on the shared virtiofs so every VM hits the same downloaded
# artifacts (SDXL alone is 13 GB). mlcr defaults to $HOME/MLC/, which is on the
# VM's local disk -- without this, each VM would re-download everything.
# We symlink $HOME/MLC -> $BASE_PATH/MLC instead of changing config, so mlcr's
# internal absolute paths (which are baked into cached metadata) keep working.
MLC_SHARED="$BASE_PATH/MLC"
mkdir -p "$MLC_SHARED"
if [ -d "$HOME/MLC" ] && [ ! -L "$HOME/MLC" ]; then
    # Migrate a prior local-disk cache into the share, then drop the empty dirs.
    rsync -a --remove-source-files "$HOME/MLC/" "$MLC_SHARED/"
    find "$HOME/MLC" -depth -type d -empty -delete 2>/dev/null || true
fi
[ -L "$HOME/MLC" ] || ln -s "$MLC_SHARED" "$HOME/MLC"

"$HOME/.local/bin/mlcr" install,python-venv --name=mlperf
export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"

# Download artifacts only -- the run-mlperf wrapper also runs the inference
# benchmark even with --actions=download,preprocess, so call the lower-level
# `get,ml-model` / `get,dataset` / `get,preprocessed-dataset` scripts directly.
#
# stdin: mlcr DOES prompt when multiple cached outputs exist (e.g. two python3
# venvs after install,python-venv). We feed it empty lines (= "press Enter,
# pick option 0") via process substitution rather than a `yes "" |` pipe -- a
# pipe would die under `set -o pipefail` when `yes` gets SIGPIPE after mlcr
# returns. Process substitution puts `yes` in a separate process whose exit
# status doesn't propagate to the parent pipeline.

# SDXL model.
"$HOME/.local/bin/mlcr" get,ml-model,sdxl,_pytorch        < <(yes "")

# coco2014 validation set: the get-dataset-coco2014 script clones master of
# mlcommons/inference and runs text_to_image/tools/coco.py, which uses PEP 604
# `str | None` syntax (Python 3.10+). Focal ships Python 3.8 only. Pre-clone the
# repo via mlcr, patch the file, then run the dataset download which sees the
# cached (patched) clone.
"$HOME/.local/bin/mlcr" get,git,repo,_branch.master,_repo.https://github.com/mlcommons/inference < <(yes "")
find "$HOME/MLC/repos/local/cache" -path '*text_to_image/tools/coco.py' \
    -exec sed -i 's/str | None = None/str = None/g' {} +
"$HOME/.local/bin/mlcr" get,dataset,coco2014,_validation  < <(yes "")

# ResNet50: ONNX model + preprocessed ImageNet validation set.
# `image-classification` disambiguates the resnet50 model script from
# get-ml-model-abtf-ssd-pytorch (which also has a `resnet50` tag).
# Preprocessed-imagenet's actual tags are get,dataset,imagenet,preprocessed --
# not get,preprocessed-dataset,imagenet. NCHW is the default layout, so no
# variant suffix needed.
"$HOME/.local/bin/mlcr" get,ml-model,resnet50,image-classification,_onnx  < <(yes "")
"$HOME/.local/bin/mlcr" get,dataset,imagenet,preprocessed                 < <(yes "")
