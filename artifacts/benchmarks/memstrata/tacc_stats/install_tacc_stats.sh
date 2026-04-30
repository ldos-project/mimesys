tacc_stats_path="$HOME/HPCPerfStats/monitor"

orig_dir=$(pwd)
sudo apt-get update
sudo apt-get install -y \
  gcc g++ libacl1-dev libaio-dev libapparmor-dev libatomic1 libattr1-dev \
  libbsd-dev libcap-dev libeigen3-dev libgbm-dev libcrypt-dev libglvnd-dev \
  libipsec-mb-dev libjpeg-dev libjudy-dev libkeyutils-dev libkmod-dev libmd-dev \
  libmpfr-dev libsctp-dev libxxhash-dev zlib1g-dev netcat supervisor rsync \
  syslog-ng vim net-tools lsof pigz libmysqlclient-dev libpq-dev autoconf \
  automake libtool gettext librabbitmq-dev rabbitmq-server libibmad-dev \
  libibumad-dev libev-dev pkg-config libsystemd-dev zip

cd ~
git clone https://github.com/TACC/HPCPerfStats.git

cd $tacc_stats_path
autoreconf --install
autoconf
./configure --disable-rabbitmq --disable-lustre
make
cp $orig_dir/tacc_stats/stats.x $tacc_stats_path/src/stats.x
sudo make install
sudo modprobe msr
sudo modprobe cpuid
sudo modprobe ib_core
