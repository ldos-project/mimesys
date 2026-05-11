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

# Mount the host-side virtiofs share at BASE_PATH so a single VM's install
# populates /dev/shm/shared on the host and every other VM sees the artifacts
# via the same share. Idempotent via the findmnt guard.
#
# Why ctypes mount(2) instead of `mount -t virtiofs shared $BASE_PATH`:
# util-linux 2.34 (Ubuntu 20.04 / focal) mis-resolves the virtiofs source tag
# and ends up calling mount(2) with the target path as the source, which the
# kernel rejects with EINVAL ("tag </home/ubuntu/shared> not found"). Calling
# the syscall directly with source="shared" works.
if ! findmnt -t virtiofs "$BASE_PATH" >/dev/null 2>&1; then
    sudo python3 -c "
import ctypes, os, sys
libc = ctypes.CDLL('libc.so.6', use_errno=True)
rc = libc.mount(b'shared', b'$BASE_PATH', b'virtiofs', 0, None)
if rc != 0:
    sys.exit('mount(2) virtiofs shared -> $BASE_PATH failed: ' + os.strerror(ctypes.get_errno()))
"
fi
cd "$BASE_PATH"

# Most stacks rely on Pond's prebuilt artifacts; clone once.
if [ ! -d "Private-Pond" ]; then
    git clone https://github.com/MoatLab/Pond.git Private-Pond
    (cd Private-Pond && git checkout e9ae753669f98497f36c9ba52525e062515c5bf1)
fi

sudo apt-get update
