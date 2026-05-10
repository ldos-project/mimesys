#!/bin/bash
# Install FASTER + Redis + Memcached + YCSB driver.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y \
    libtbb-dev build-essential pkg-config \
    libaio-dev libaio1 uuid-dev libnuma-dev cmake \
    default-jre default-jdk openjdk-8-jre openjdk-8-jdk \
    ssh pdsh bc scala python2 memcached

# FASTER
cd "$BASE_PATH"
if [ ! -d "FASTER" ]; then
    git clone https://github.com/yuhong-zhong/FASTER.git
fi
mkdir -p FASTER/cc/build/Release
cd FASTER/cc/build/Release
cmake -DCMAKE_BUILD_TYPE=Release ../..
make pmem_benchmark

# Redis
cd "$BASE_PATH"
if [ ! -d "redis" ]; then
    wget https://github.com/redis/redis/archive/refs/tags/7.2.3.tar.gz
    tar -xzvf 7.2.3.tar.gz
    mv redis-7.2.3 redis
fi
cd redis && make -j8
sudo bash -c "echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf"
sed -i -e '$a save ""' redis.conf

# Maven (for YCSB)
cd "$BASE_PATH"
if [ ! -d "apache-maven" ]; then
    wget https://downloads.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz
    tar -xzvf apache-maven-3.9.11-bin.tar.gz
    mv apache-maven-3.9.11 apache-maven
fi
echo "export M2_HOME=$BASE_PATH/apache-maven" >> ~/.bashrc
echo 'export PATH=$PATH:$M2_HOME/bin'         >> ~/.bashrc
export M2_HOME="$BASE_PATH/apache-maven"
export PATH="$PATH:$M2_HOME/bin"

# YCSB
cd "$BASE_PATH"
if [ ! -d "YCSB" ]; then
    git clone https://github.com/brianfrankcooper/YCSB.git
fi
cd YCSB
mvn -pl site.ycsb:redis-binding     -am clean package
mvn -pl site.ycsb:memcached-binding -am clean package
