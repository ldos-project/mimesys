#!/bin/bash
# SSH / virsh helpers shared by the runner and CLI.
# Source after lib/host.env so SSH_OPTS is defined.

# Print the IPv4 address of a libvirt domain (empty if not yet leased).
vm_ip() {
    local dom="$1"
    sudo virsh domifaddr "$dom" | awk '/vnet/ {sub(/\/.*/, "", $4); print $4}'
}

# Block until $dom has an IP and SSH answers.
vm_wait_ready() {
    local dom="$1"
    local ip
    ip="$(vm_ip "$dom")"
    while [ -z "$ip" ]; do
        sleep 1
        ip="$(vm_ip "$dom")"
    done
    while true; do
        local out
        out="$(sudo ssh $SSH_OPTS ubuntu@$ip 'echo ok' 2>/dev/null)" || true
        if [ "$out" = "ok" ]; then
            return 0
        fi
        sleep 1
    done
}

# Graceful shutdown with timeout, then destroy.
vm_kill() {
    local dom="$1"
    if [ -z "$(pgrep -f "$dom")" ]; then
        return 0
    fi
    sudo virsh shutdown "$dom" || true
    local count=0
    while [ -n "$(pgrep -f "$dom")" ]; do
        sleep 10
        count=$((count + 1))
        if [ "$count" -eq 12 ]; then
            sudo virsh destroy "$dom" || true
            break
        fi
    done
}

# scp to ubuntu@<dom-ip>
vm_scp_to() {
    local dom="$1" src="$2" dst="$3"
    local ip
    ip="$(vm_ip "$dom")"
    scp $SSH_OPTS "$src" ubuntu@"$ip":"$dst"
}

# scp from ubuntu@<dom-ip>
vm_scp_from() {
    local dom="$1" src="$2" dst="$3"
    local ip
    ip="$(vm_ip "$dom")"
    scp $SSH_OPTS ubuntu@"$ip":"$src" "$dst"
}

# Run a command inside a VM.
vm_ssh() {
    local dom="$1"; shift
    local ip
    ip="$(vm_ip "$dom")"
    ssh $SSH_OPTS ubuntu@"$ip" "$@"
}
