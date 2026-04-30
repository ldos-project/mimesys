#!/bin/bash
set -eux -o pipefail

######################################################
#
#  Script input parameters
#
######################################################

exp_arr=($1)
numa_node=$2
measure_emon=false

cd ${exp_arr[0]}
rm -f result*
cd -

######################################################
#
#  VM parameters
#
######################################################

num_vms=${#exp_arr[@]}
num_vcpus_arr=()
vm_core_arr=()
rss_MB_arr=()
vm_size_MB_arr=()

core_base=0
for exp in "${exp_arr[@]}"; do
	source $exp/exp_config.sh
	num_vcpus_arr+=($num_cores)
	rss_MB_arr+=($rss_MB)
	vm_size_MB_arr+=($vm_size_MB)

	vm_core=""
	for core_index in $(seq 1 $num_cores); do
		if [ "$core_index" != "1" ]; then
			vm_core+=","
		fi
		vm_core+="$(( $core_base + $core_index - 1 ))"
	done
	vm_core_arr+=($vm_core)
	core_base=$(( $core_base + $num_cores ))
done

######################################################
#
#  Initialize system
#
######################################################

page_shift=21

uvt-kvm() {
	case $1 in
		ip)
			sudo virsh domifaddr $2 | awk '/vnet/ {sub(/\/.*/, "", $4); print $4}'
			;;
		ssh)
			ip="$(sudo virsh domifaddr $2 | awk '/vnet/ {sub(/\/.*/, "", $4); print $4}')"
			sudo ssh ubuntu@$ip
			;;
		wait)
			ip="$(sudo virsh domifaddr $2 | awk '/vnet/ {sub(/\/.*/, "", $4); print $4}')"
			while [ -z "$ip" ]; do
				ip="$(sudo virsh domifaddr $2 | awk '/vnet/ {sub(/\/.*/, "", $4); print $4}')"
				sleep 1
			done
			while true; do
				output="$(sudo ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip 'echo succeed' 2> /dev/null)" || true
				if [ "$output" != "succeed" ]; then
					sleep 1
				else
					break
				fi
			done
			;;
		*)
			echo "Invalid argument $1"
			;;
	esac
}

echo 1800 > /proc/sys/kernel/panic
echo > /sys/kernel/debug/tracing/trace
echo 0 > /proc/sys/kernel/numa_balancing
echo 2 > /sys/kernel/mm/ksm/run
echo 1 > /proc/sys/vm/drop_caches

while [ ! -z "$(pgrep -fn 'collect-edp')" ]; do
	kill $(pgrep -fn 'collect-edp')
	sleep 1
done

# Kill all VMs
for vm_idx in $(seq 1 $num_vms); do
	count=0
	if [ -z "$(pgrep -f vm${vm_idx})" ]; then
		continue
	fi
	sudo virsh shutdown vm${vm_idx} || true
	while [ ! -z "$(pgrep -f vm${vm_idx})" ]; do
		sleep 10
		count=$(( $count + 1 ))
		if [ "$count" -eq "12" ]; then
			sudo virsh destroy vm${vm_idx} || true
			break
		fi
	done
done

# Disable CPU frequency scaling
SCRIPT_PATH=`realpath $0`
BASE_DIR=`dirname $SCRIPT_PATH`
bash "$BASE_DIR/utils/disable_cpu_freq_scaling.sh"

######################################################
#
#  Start VMs
#
######################################################

for vm_idx in $(seq 1 $num_vms); do
	rss_MB=${rss_MB_arr[$(($vm_idx - 1))]}
	vm_size_MB=${vm_size_MB_arr[$(($vm_idx - 1))]}
	num_vcpus=${num_vcpus_arr[$(($vm_idx - 1))]}
	vm_core=${vm_core_arr[$(($vm_idx - 1))]}

	memory_size_MB=$vm_size_MB
	vm_mem_MB_arr+=($memory_size_MB)

	virsh dumpxml vm$vm_idx > /tmp/vm$vm_idx.xml

	xmlstarlet ed --inplace -u "/domain/cpu" -x "string(.)" /tmp/vm$vm_idx.xml
	xmlstarlet ed --inplace -d "/domain/numatune" /tmp/vm$vm_idx.xml

	xmlstarlet ed --inplace -s "/domain/cpu" -t elem -n numa -v ""  /tmp/vm$vm_idx.xml
	xmlstarlet ed --inplace -s "/domain/cpu/numa" -t elem -n cell -v "" \
		-i //numa/cell -t attr -n id -v "0" \
		-i //numa/cell -t attr -n cpus -v "0-$(($num_vcpus - 1))" \
		-i //numa/cell -t attr -n memory -v "$(($memory_size_MB * 1024))" \
		-i //numa/cell -t attr -n unit -v "KiB" /tmp/vm$vm_idx.xml
	xmlstarlet ed --inplace -s "/domain" -t elem -n numatune -v "" \
		-s /domain/numatune -t elem -n memnode -v "" \
		-i /domain/numatune/memnode -t attr -n cellid -v "0" \
		-i /domain/numatune/memnode -t attr -n mode -v "strict" \
		-i /domain/numatune/memnode -t attr -n nodeset -v "$numa_node" /tmp/vm$vm_idx.xml

	sed -i "s|<memory.*</memory>|<memory unit='KiB'>$(($memory_size_MB * 1024))</memory>|" /tmp/vm$vm_idx.xml
	sed -i "s|<currentMemory.*</currentMemory>|<currentMemory unit='KiB'>$(($memory_size_MB * 1024))</currentMemory>|" /tmp/vm$vm_idx.xml
	sed -i "s|<vcpu.*</vcpu>|<vcpu placement='static' cpuset='$vm_core'>$num_vcpus</vcpu>|" /tmp/vm$vm_idx.xml
	virsh define /tmp/vm$vm_idx.xml --validate

	virsh start vm$vm_idx
	uvt-kvm wait vm$vm_idx
done

######################################################
#
#  Run both workloads and the orchestrator
#
######################################################

vm_ip_arr=()
vm_pid_arr=()
for vm_idx in $(seq 1 $num_vms); do
	vm_ip_arr+=($(uvt-kvm ip vm$vm_idx))
	vm_pid_arr+=($(pgrep -f vm$vm_idx))
done

for vm_idx in $(seq 1 $num_vms); do
	vm_ip=${vm_ip_arr[$(($vm_idx - 1))]}
	exp=${exp_arr[$(($vm_idx - 1))]}

	scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $exp/prepare_exp.sh \
		ubuntu@$vm_ip:/home/ubuntu/prepare_exp.sh
	scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $exp/run_exp.sh \
		ubuntu@$vm_ip:/home/ubuntu/run_exp.sh

	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
	"cd /home/ubuntu/; mkdir -p shared; cd shared; sudo mount -t virtiofs shared /home/ubuntu/shared"
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
	"cd /home/ubuntu/; bash /home/ubuntu/shared/create_symbolic_links.sh || true"
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
    	"cd /home/ubuntu/; source ~/.bashrc; bash ./prepare_exp.sh"
done

# Measure using emon
if [ "$measure_emon" = true ]; then
	emon -collect-edp -f result_emon.dat &
	emon_pid=$!
fi

if [ -z "$LOCK_FILE_PATH" ]; then
	echo "LOCK_FILE_PATH is not set. Skipping."
else
	echo "1" > "$LOCK_FILE_PATH"
fi

tacc_stats_dir="/users/dhkim/HPCPerfStats/monitor/src"

# start
sudo $tacc_stats_dir/hpcperfstatsd begin
# Start a background process that echoes every 1s
(
	while true; do
		sudo $tacc_stats_dir/hpcperfstatsd collect
		sleep 1
	done
) &
collect_pid=$!

# Start a background process for power monitoring
sudo bash -c '
	LOGFILE="/var/log/hpcperfstats/power_monitor.log"
	while true; do
		tstamp=$(date +%s.%6N)

		total_uj=0
		for f in /sys/class/powercap/*/energy_uj; do
			if [[ -r "$f" ]]; then
				val=$(cat "$f")
				total_uj=$((total_uj + val))
			fi
		done

		total_j=$(awk "BEGIN {printf \"%.6f\", $total_uj/1000000}")

		echo "$tstamp,$total_j" >> "$LOGFILE"

		sleep 1
	done
' &
power_pid=$!

# Run workloads
for vm_idx in $(seq 1 $num_vms); do
	vm_ip=${vm_ip_arr[$(($vm_idx - 1))]}

	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip \
    	"cd /home/ubuntu/; source ~/.bashrc; bash ./run_exp.sh > /home/ubuntu/result_run_exp_stdout.txt 2> /home/ubuntu/result_run_exp_stderr.txt < /dev/null &"
done

######################################################
#
#  Wait for both workloads to finish
#
######################################################

for vm_idx in $(seq 1 $num_vms); do
	exp=${exp_arr[$(($vm_idx - 1))]}
	vm_ip=${vm_ip_arr[$(($vm_idx - 1))]}
	while [ ! -z "$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$vm_ip "pgrep -f run_exp")" ]; do
		sleep 30
	done
done

if [ "$measure_emon" = true ]; then
	kill -2 $emon_pid
fi

# kill the background process
sudo kill $collect_pid
sudo kill $power_pid

######################################################
#
#  Collect results
#
######################################################

for vm_idx in $(seq 1 $num_vms); do
	exp=${exp_arr[$(($vm_idx - 1))]}
	vm_ip=${vm_ip_arr[$(($vm_idx - 1))]}
	exp_name=$(basename "$exp")
	scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
		ubuntu@$vm_ip:/home/ubuntu/result_app_perf.txt result_app_perf_${vm_idx}_${exp_name}.txt
done

######################################################
#
#  Cleanup
#
######################################################

for vm_idx in $(seq 1 $num_vms); do
	count=0
	sudo virsh shutdown vm${vm_idx}
	while [ ! -z "$(pgrep -f vm${vm_idx})" ]; do
		sleep 10
		count=$(( $count + 1 ))
		if [ "$count" -eq "12" ]; then
			sudo virsh destroy vm${vm_idx} || true
			break
		fi
	done
done
