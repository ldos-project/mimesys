HOME_PATH=${HOME_PATH:-/users/dhkim}

cd $HOME_PATH
git clone git@github.com:kdh0102/memstrata.git
cd memstrata

WORKLOAD_PATH=${WORKLOAD_PATH:-/dev/shm}
mkdir $WORKLOAD_PATH/shared
mkdir $WORKLOAD_PATH/shared/MLC
# Install Workloads in VMs
cp create_symbolic_links.sh $WORKLOAD_PATH/shared
cp vm_install_workloads.sh $WORKLOAD_PATH/shared

# Run these in sequential due to apt lock issue
for vm in vm1 vm2 vm3 vm4 vm5 vm6 vm7 vm8 vm9 vm10; do
sudo virsh shutdown "$vm" || true
  # sudo bash create_vm.sh "$vm"
done

# VMs for web
for vm in vm9 vm10; do
  if [ "$vm" = "vm1" ] || [ "$vm" = "vm2" ] || [ "$vm" = "vm3" ]; then
    workload_type="web"
  fi
  if [ "$vm" = "vm4" ] || [ "$vm" = "vm5" ] || [ "$vm" = "vm6" ]; then
    workload_type="bigdata"
  fi
  if [ "$vm" = "vm6" ]; then
    workload_type="database"
  fi
  if [ "$vm" = "vm7" ] || [ "$vm" = "vm8" ]; then
    workload_type="kvstore"
  fi
  if [ "$vm" = "vm9" ]; then
    workload_type="ml"
  fi
  if [ "$vm" = "vm10" ]; then
    workload_type="graph"
  fi
  echo "Installing workloads in $vm"
  vm=vm2
  vm_ip=$(sudo virsh domifaddr "$vm" | awk '/ipv4/ {print $4}' | cut -d'/' -f1)
  sudo ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
    "cd /home/ubuntu/; mkdir -p shared; sudo chown ubuntu:ubuntu shared; cd shared; sudo mount -t virtiofs shared /home/ubuntu/shared"
  sudo ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
	"cd /home/ubuntu/; bash /home/ubuntu/shared/create_symbolic_links.sh || true"
  sudo ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
	"cd /home/ubuntu/; sudo chown ubuntu:ubuntu /home/ubuntu/shared/MLC"
  sudo ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
    "/home/ubuntu/shared/vm_install_workloads.sh $workload_type"
done
#
# # Disable CPU freq scaling and hyperthreading
# bash utils/disable_cpu_freq_scaling.sh

cd $HOME_PATH/fleetbench
# bash collect_mimesys_data.sh
