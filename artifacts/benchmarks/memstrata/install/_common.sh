#!/bin/bash
# Common setup shared by every install/*.sh module.
# Runs inside a VM.  Idempotent.
set -eu -o pipefail

# Quieter bashrc + non-color prompt so SSH heredocs stay sane.
sed -i '/^case $- in$/,/^esac$/d' "$HOME/.bashrc" || true
grep -q '^force_color_prompt=""' "$HOME/.bashrc" 2>/dev/null \
    || sed -i '1i force_color_prompt=""' "$HOME/.bashrc"
grep -q '^color_prompt=""' "$HOME/.bashrc" 2>/dev/null \
    || sed -i '1i color_prompt=""' "$HOME/.bashrc"

export BASE_PATH="$HOME/shared"
mkdir -p "$BASE_PATH"
cd "$BASE_PATH"

# Most stacks rely on Pond's prebuilt artifacts; clone once.
if [ ! -d "Private-Pond" ]; then
    git clone https://github.com/MoatLab/Pond.git Private-Pond
    (cd Private-Pond && git checkout e9ae753669f98497f36c9ba52525e062515c5bf1)
fi

sudo apt-get update
