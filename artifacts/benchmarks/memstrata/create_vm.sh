#!/bin/bash
set -eux -o pipefail

sudo apt-get update
sudo apt-get install libusbredirparser-dev libusb-1.0-0-dev ninja-build git libusb-dev libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev git-email libaio-dev libbluetooth-dev libbrlapi-dev libbz2-dev libcap-dev libcap-ng-dev libcap-dev libcap-ng-dev libcurl4-gnutls-dev libgtk-3-dev librbd-dev librdmacm-dev libsasl2-dev libsdl1.2-dev libseccomp-dev libsnappy-dev libssh2-1-dev libvde-dev libvdeplug-dev libxen-dev liblzo2-dev valgrind xfslibs-dev libnfs-dev libiscsi-dev -y
sudo apt update
sudo apt install -y xmlstarlet htop uvtool libvirt-daemon-system libvirt-clients bridge-utils virt-manager libosinfo-bin libguestfs-tools virt-top

sudo xmlstarlet ed --inplace -d '//graphics' /usr/share/uvtool/libvirt/template.xml
sudo xmlstarlet ed --inplace -d '//video' /usr/share/uvtool/libvirt/template.xml

if [ ! -f /var/lib/libvirt/images/base/focal-server-cloudimg-amd64.img ]; then
    wget https://cloud-images.ubuntu.com/focal/current/focal-server-cloudimg-amd64.img
    sudo mkdir /var/lib/libvirt/images/base
    sudo mv focal-server-cloudimg-amd64.img /var/lib/libvirt/images/base/
fi

domain_name="$1"

sudo mkdir -p /var/lib/libvirt/images/$domain_name
sudo qemu-img create -f qcow2 -F qcow2 -o \
    backing_file=/var/lib/libvirt/images/base/focal-server-cloudimg-amd64.img \
    /var/lib/libvirt/images/$domain_name/$domain_name.qcow2
sudo qemu-img resize /var/lib/libvirt/images/$domain_name/$domain_name.qcow2 225G

cd /tmp
cat >meta-data <<EOF
local-hostname: $domain_name
EOF

export PUB_KEY=$(sudo cat /root/.ssh/id_rsa.pub)
if [ -z "$PUB_KEY" ]; then
    echo "Cannot find the public key of root"
    exit 1
fi

cat >user-data <<EOF
#cloud-config
users:
  - name: ubuntu
    ssh-authorized-keys:
      - $PUB_KEY
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    groups: sudo
    shell: /bin/bash
runcmd:
  - echo "AllowUsers ubuntu" >> /etc/ssh/sshd_config
  - restart ssh
EOF

sudo genisoimage -output /var/lib/libvirt/images/$domain_name/$domain_name-cidata.iso -volid cidata -joliet -rock user-data meta-data

sudo virt-install --connect qemu:///system --virt-type kvm --name $domain_name \
    --ram $((32 * 1024)) --vcpus=8 --os-variant ubuntu20.04 \
    --disk path=/var/lib/libvirt/images/$domain_name/$domain_name.qcow2,format=qcow2 \
    --disk /var/lib/libvirt/images/$domain_name/$domain_name-cidata.iso,device=cdrom \
    --memorybacking source.type=memfd,access.mode=shared \
    --filesystem source=/dev/shm/shared,target=shared,type=mount,accessmode=passthrough,driver.type=virtiofs \
    --import --network network=default --noautoconsole

sudo uvt-kvm wait $domain_name --insecure

# Disable cache for the disk
sudo virsh dumpxml $domain_name > /tmp/$domain_name.xml
sudo xmlstarlet ed --inplace -s "/domain/devices/disk/driver" -t attr -n "cache" -v "none" /tmp/$domain_name.xml
sudo virsh define /tmp/$domain_name.xml --validate

sudo virsh reboot $domain_name
sudo uvt-kvm wait $domain_name --insecure
