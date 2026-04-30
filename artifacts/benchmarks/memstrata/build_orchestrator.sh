#!/bin/bash
set -eux -o pipefail

if [ "$(uname -r)" !=  "5.19.0-memstrata+" ]; then
    printf "Not in Memstrata kernel. Please run the following commands to boot into Memstrata kernel:\n"
    printf "    sudo grub-reboot \"Advanced options for Ubuntu>Ubuntu, with Linux 5.19.0-memstrata+\"\n"
    printf "    sudo reboot\n"
    exit 1
fi

SCRIPT_PATH=`realpath $0`
BASE_DIR=`dirname $SCRIPT_PATH`
ORCHESTRATOR_PATH="$BASE_DIR/orchestrator"

# Install OXXN
sudo apt install python3-pip -y
sudo python3 -m pip install onnxruntime

cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xvf onnxruntime-linux-x64-1.16.3.tgz
cd onnxruntime-linux-x64-1.16.3
cd include
sudo cp * /usr/include/
cd ../lib
sudo cp * /usr/lib64/
sudo cp * /usr/lib/

# Install dependencies
sudo apt-get update
sudo apt-get install libnuma-dev cmake build-essential -y

# Build orchestrator
pushd $ORCHESTRATOR_PATH
rm -rf build
mkdir build
cd build
cmake ..
make -j8
popd
