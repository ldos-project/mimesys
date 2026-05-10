#!/bin/bash
# Rewrite a libvirt domain XML so it has the requested vCPU pinning, memory,
# and a single-cell NUMA topology that matches the VM's vCPU/mem footprint.
#
# We intentionally do NOT emit a <numatune> element binding the VM to a host
# NUMA node -- that knob is stale legacy from the upstream memstrata repo
# (user confirmed it's unused).  Memory just lands wherever the host scheduler
# puts it.
#
# Usage: vm_define_pinned <dom> <num_vcpus> <vm_core_csv> <memory_size_MB>
vm_define_pinned() {
    local dom="$1" num_vcpus="$2" vm_core="$3" mem_MB="$4"
    local xml="/tmp/${dom}.xml"

    sudo virsh dumpxml "$dom" > "$xml"

    # Clear any existing inline <cpu> children and any prior numatune.
    xmlstarlet ed --inplace -u "/domain/cpu" -x "string(.)" "$xml"
    xmlstarlet ed --inplace -d "/domain/numatune" "$xml"

    # Re-add a single-cell <cpu><numa><cell .../></numa></cpu> matching this VM.
    xmlstarlet ed --inplace -s "/domain/cpu" -t elem -n numa -v "" "$xml"
    xmlstarlet ed --inplace \
        -s "/domain/cpu/numa" -t elem -n cell -v "" \
        -i "//numa/cell" -t attr -n id -v "0" \
        -i "//numa/cell" -t attr -n cpus -v "0-$((num_vcpus - 1))" \
        -i "//numa/cell" -t attr -n memory -v "$((mem_MB * 1024))" \
        -i "//numa/cell" -t attr -n unit -v "KiB" "$xml"

    # Update top-level memory + vcpu pinning.
    sed -i "s|<memory.*</memory>|<memory unit='KiB'>$((mem_MB * 1024))</memory>|" "$xml"
    sed -i "s|<currentMemory.*</currentMemory>|<currentMemory unit='KiB'>$((mem_MB * 1024))</currentMemory>|" "$xml"
    sed -i "s|<vcpu.*</vcpu>|<vcpu placement='static' cpuset='$vm_core'>$num_vcpus</vcpu>|" "$xml"

    sudo virsh define "$xml" --validate >/dev/null
}

# Build a comma-separated CPU pinning list of $num_cores starting at $core_base.
# Echo it on stdout.
vm_pin_cores() {
    local num_cores="$1" core_base="$2"
    local out=""
    for i in $(seq 0 $((num_cores - 1))); do
        if [ -n "$out" ]; then out+=","; fi
        out+="$((core_base + i))"
    done
    printf "%s" "$out"
}
