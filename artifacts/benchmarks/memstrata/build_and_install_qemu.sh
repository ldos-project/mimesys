#!/bin/bash
set -eux -o pipefail

if [ "$(uname -r)" !=  "5.19.0-memstrata+" ]; then
    printf "Not in Memstrata kernel. Please run the following commands to boot into Memstrata kernel:\n"
    printf "    sudo grub-reboot \"Advanced options for Ubuntu>Ubuntu, with Linux 5.19.0-memstrata+\"\n"
    printf "    sudo reboot\n"
    exit 1
fi

# Install build dependencies
sudo apt-get update
sudo apt-get install libusbredirparser-dev libusb-1.0-0-dev ninja-build git libusb-dev libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev git-email libaio-dev libbluetooth-dev libbrlapi-dev libbz2-dev libcap-dev libcap-ng-dev libcap-dev libcap-ng-dev libcurl4-gnutls-dev libgtk-3-dev librbd-dev librdmacm-dev libsasl2-dev libsdl1.2-dev libseccomp-dev libsnappy-dev libssh2-1-dev libvde-dev libvdeplug-dev libxen-dev liblzo2-dev valgrind xfslibs-dev libnfs-dev libiscsi-dev -y

SCRIPT_PATH=`realpath $0`
BASE_DIR=`dirname $SCRIPT_PATH`
QEMU_PATH="$BASE_DIR/qemu"

# Build and install QEMU
pushd $QEMU_PATH
git submodule init
git submodule update --recursive

rm -rf build
mkdir build
cd build
../configure --target-list=x86_64-softmmu --enable-debug
make -j8
sudo make install
popd

# Install libvirt
sudo apt install cpu-checker -y
sudo apt install -y htop uvtool libvirt-daemon-system libvirt-clients bridge-utils virt-manager libosinfo-bin libguestfs-tools virt-top

# Modify uvt-kvm template file
sudo apt update
sudo apt install xmlstarlet -y
sudo xmlstarlet ed --inplace -d '//graphics' /usr/share/uvtool/libvirt/template.xml
sudo xmlstarlet ed --inplace -d '//video' /usr/share/uvtool/libvirt/template.xml

# Download image
wget https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-amd64.img
sudo mkdir /var/lib/libvirt/images/base
sudo mv focal-server-cloudimg-amd64.img /var/lib/libvirt/images/base/
