HOME_PATH=${HOME_PATH:-$HOME}

# ─── Progress / status tracking ────────────────────────────────────────────
# Writes a single-line status to $STATUS_FILE so install_remote_dependencies.py
# (and `uv run python install_remote_dependencies.py --status`) can tell at a
# glance whether the install is still running, has finished cleanly, or has
# failed. STATUS_FILE values:
#   running stage=<name> pid=<pid>   — current phase
#   ok                                — finished successfully
#   failed exit=<code> stage=<name>   — last phase that errored
STATUS_FILE="$HOME_PATH/.mimesys_install.status"
LOG_FILE="$HOME_PATH/install_run.log"
CURRENT_STAGE="init"
mark_stage() { CURRENT_STAGE="$1"; echo "running stage=$CURRENT_STAGE pid=$$ ts=$(date +%s)" > "$STATUS_FILE"; }
on_exit() {
  ec=$?
  if [ $ec -eq 0 ]; then
    echo "ok ts=$(date +%s)" > "$STATUS_FILE"
  else
    echo "failed exit=$ec stage=$CURRENT_STAGE ts=$(date +%s)" > "$STATUS_FILE"
  fi
}
trap on_exit EXIT
set -e -o pipefail   # any failure aborts and the trap reports the failing stage

mark_stage init

mark_stage apt_base
sudo apt-get update
sudo apt-get install -y \
  gcc g++ libacl1-dev libaio-dev libapparmor-dev libatomic1 libattr1-dev \
  libbsd-dev libcap-dev libeigen3-dev libgbm-dev libcrypt-dev libglvnd-dev \
  libipsec-mb-dev libjpeg-dev libjudy-dev libkeyutils-dev libkmod-dev libmd-dev \
  libmpfr-dev libsctp-dev libxxhash-dev zlib1g-dev netcat supervisor rsync \
  syslog-ng vim net-tools lsof pigz libmysqlclient-dev libpq-dev autoconf \
  automake libtool gettext librabbitmq-dev rabbitmq-server libibmad-dev \
  libibumad-dev libev-dev pkg-config libsystemd-dev zip libhdf5-dev hdf5-tools \
  libnuma-dev intel-cmt-cat

mark_stage hpcperfstats_build
cd $HOME_PATH/HPCPerfStats/monitor && \
autoreconf --install && \
autoconf && \
./configure --disable-rabbitmq --disable-lustre && \
cp $HOME_PATH/stats.x $HOME_PATH/HPCPerfStats/monitor/src/stats.x && \
make && \
sudo make install
# modprobes are best-effort: depending on the kernel, the modules may already
# be built-in (no-op) or absent; neither case should abort the install.
sudo modprobe cpuid || true
sudo modprobe msr || true
sudo modprobe ib_core || true

mark_stage apt_clang
sudo apt-get update
sudo apt-get install -y clang llvm lld libpapi-dev python3-pip htop

mark_stage apt_msr_tools
sudo apt update
sudo apt install -y msr-tools
sudo modprobe msr
# sudo wrmsr -a 0x1a4 0xF    # Disable all prefetchers on all cores
# sudo wrmsr -a 0x1a4 0x0    # Enable all prefetchers on all cores
# sudo rdmsr -a 0x1a4

mark_stage bazelisk
curl -Lo bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk
sudo mv bazelisk /usr/local/bin/bazel
bazel version

mark_stage clone_mimesys
cd $HOME_PATH
if [ "${SKIP_CLONE_MIMESYS:-0}" = "1" ]; then
    # Controller has already rsync'd the local fleetbench to $HOME_PATH/fleetbench.
    if [ ! -d "$HOME_PATH/fleetbench" ]; then
        echo "ERROR: SKIP_CLONE_MIMESYS=1 set but $HOME_PATH/fleetbench is missing" >&2
        exit 1
    fi
    echo "  [clone_mimesys] using pre-staged fleetbench at $HOME_PATH/fleetbench"
else
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:ldos-project/mimesys.git
    cd mimesys
    cd $HOME_PATH
    mv mimesys/fleetbench fleetbench
fi
cd fleetbench
mkdir -p fleetbench/mimesys/execution_plans
# The mimesys BUILD globs execution_plans/** with allow_empty=False, so we
# need at least one file in here for analysis to succeed. The placeholder
# stays empty (touch) — bazel only checks that the glob isn't empty.
touch fleetbench/mimesys/execution_plans/plan_0.h5

mark_stage bazel_build
# `bazel build` (not `run`) — we only need to verify the binary compiles.
# `bazel run` would also execute the benchmark, which throws
# H5::FileIException on the empty plan_0.h5 placeholder. Real plans are run
# later via collect_mimesys_data.sh after generate_plans.py populates them.
TACC_STATS_DIR=$HOME_PATH/HPCPerfStats/monitor/src sudo bazel build --config=clang --config=opt fleetbench/mimesys:mimesys_benchmark

mark_stage generate_plans
pip install matplotlib h5py
rm fleetbench/mimesys/execution_plans/plan_0.h5
cd fleetbench/mimesys/execution_plan_generation
python generate_plans.py
cd -
mv fleetbench/mimesys/execution_plans .
mkdir fleetbench/mimesys/execution_plans
# Re-seed the BUILD-glob placeholder so later `bazel build` invocations
# succeed; the benchmark skips the empty plan_0.h5.
touch fleetbench/mimesys/execution_plans/plan_0.h5

mark_stage disable_cpu_freq_scaling
bash scripts/disable_cpu_freq_scaling.sh
