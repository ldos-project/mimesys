HOME_PATH=${HOME_PATH:-/users/dhkim}

sudo apt-get update
sudo apt-get install -y \
  gcc g++ libacl1-dev libaio-dev libapparmor-dev libatomic1 libattr1-dev \
  libbsd-dev libcap-dev libeigen3-dev libgbm-dev libcrypt-dev libglvnd-dev \
  libipsec-mb-dev libjpeg-dev libjudy-dev libkeyutils-dev libkmod-dev libmd-dev \
  libmpfr-dev libsctp-dev libxxhash-dev zlib1g-dev netcat supervisor rsync \
  syslog-ng vim net-tools lsof pigz libmysqlclient-dev libpq-dev autoconf \
  automake libtool gettext librabbitmq-dev rabbitmq-server libibmad-dev \
  libibumad-dev libev-dev pkg-config libsystemd-dev zip libhdf5-dev hdf5-tools

cd $HOME_PATH/HPCPerfStats/monitor && \
autoreconf --install && \
autoconf && \
./configure --disable-rabbitmq --disable-lustre && \
make && \
cp $HOME_PATH/stats.x $HOME_PATH/HPCPerfStats/monitor/src/stats.x && \
sudo make install && \
sudo modprobe cpuid && \
sudo modprobe msr && \
sudo modprobe ib_core

# fleetbench
sudo apt-get update
sudo apt-get install -y clang llvm lld libpapi-dev python3-pip htop

# For hardware prefetchers
sudo apt update
sudo apt install -y msr-tools
sudo modprobe msr
# sudo wrmsr -a 0x1a4 0xF    # Disable all prefetchers on all cores
# sudo wrmsr -a 0x1a4 0x0    # Enable all prefetchers on all cores
# sudo rdmsr -a 0x1a4

curl -Lo bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk
sudo mv bazelisk /usr/local/bin/bazel
bazel version

HOME_PATH=${HOME_PATH:-/users/USERNAME}
cd $HOME_PATH
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:ldos-project/llm-app-generation.git
cd llm-app-generation && git checkout refactoring # TODO: remove this after refactoring is done

cd $HOME_PATH
mv llm-app-generation/fleetbench fleetbench
cd fleetbench
mkdir fleetbench/mimesys/execution_plans
touch fleetbench/mimesys/execution_plans/plan_0.h5

TACC_STATS_DIR=$HOME_PATH/HPCPerfStats/monitor/src sudo bazel run --config=clang --config=opt fleetbench/mimesys:mimesys_benchmark -- --benchmark_filter="BM_Mimesys"

pip install matplotlib h5py
rm fleetbench/mimesys/execution_plans/plan_0.h5
cd fleetbench/mimesys/execution_plan_generation
python generate_plans.py
cd -
mv fleetbench/mimesys/execution_plans .
mkdir fleetbench/mimesys/execution_plans

# Disable CPU hyperthreading
bash scripts/disable_cpu_freq_scaling.sh
