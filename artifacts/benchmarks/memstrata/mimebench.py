#!/usr/bin/env python3
"""mimebench: CLI for running mixed-workload VM experiments.

Subcommands map 1:1 onto the five capabilities the refactor targets:

    vms create     -- spin up libvirt VMs by name
    install        -- run install/<stack>.sh inside a VM via SSH
    run-mix        -- run N workloads in N VMs in parallel, sample every K s
    run-target     -- like run-mix but waits only for the target workload to
                      finish, leaving the others as noisy neighbors
    run-batch      -- iterate over exp_configs/<node>/<N>/config.txt mixes and
                      run-target for each one (replaces generate_noisy_neighbor_exps*.py)

All scheduling, sampling, and VM lifecycle is delegated to lib/run_mix.sh so
the Python layer stays thin.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
INSTALL = ROOT / "install"
REGISTRY = ROOT / "config" / "workloads.tsv"
HOST_ENV = ROOT / "config" / "host.env"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, dict[str, str]]:
    """Return {workload_name: {"type": ..., "install_stack": ...}}."""
    out: dict[str, dict[str, str]] = {}
    for line in REGISTRY.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, wtype, stack = parts
        out[name] = {"type": wtype, "install_stack": stack}
    return out


def vm_pool_for_type(wtype: str) -> list[int]:
    """Read VM_POOL_<TYPE> from host.env (or current env)."""
    raw = os.environ.get(f"VM_POOL_{wtype.upper()}")
    if raw is None:
        # Fall back to parsing host.env directly.
        for line in HOST_ENV.read_text().splitlines():
            line = line.strip()
            key = f"VM_POOL_{wtype.upper()}"
            if line.startswith(f': "${{{key}:='):
                raw = line.split(":=", 1)[1].rstrip('}"')
                break
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def read_num_cores(workload: str) -> int:
    cfg = ROOT / "workload_scripts" / workload / "exp_config.sh"
    for line in cfg.read_text().splitlines():
        if line.strip().startswith("num_cores="):
            return int(line.split("=", 1)[1])
    raise ValueError(f"num_cores not found in {cfg}")


# ---------------------------------------------------------------------------
# Shell-out helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = True, env: dict | None = None) -> int:
    print("+", " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, env=env).returncode


def run_mix(
    workloads: Iterable[str],
    vm_ids: Iterable[int],
    *,
    wait_policy: str = "all",
    target_idx: int | None = None,
    core_base: int = 0,
    neighbor_mode: str = "none",
    lock_file: str | None = None,
    sample_interval: int = 1,
    output_dir: str | Path | None = None,
    extra_env: dict | None = None,
) -> int:
    cmd = [
        "sudo", "bash", str(LIB / "run_mix.sh"),
        "--workloads",       " ".join(workloads),
        "--vm-ids",          " ".join(str(v) for v in vm_ids),
        "--core-base",       str(core_base),
        "--wait-policy",     wait_policy,
        "--neighbor-mode",   neighbor_mode,
        "--sample-interval", str(sample_interval),
    ]
    if target_idx is not None:
        cmd += ["--target-idx", str(target_idx)]
    if lock_file:
        cmd += ["--lock-file", lock_file]
    if output_dir:
        cmd += ["--output-dir", str(output_dir)]

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return run(cmd, env=env)


# ---------------------------------------------------------------------------
# Subcommand: init resize-disk
# ---------------------------------------------------------------------------

def cmd_init_resize_disk(args: argparse.Namespace) -> int:
    """Grow the root partition + filesystem to fill the disk.

    On CloudLab images the root filesystem ships sized for the image, not for
    the actual node disk, so partition 3 has to be deleted and recreated to
    span the rest of the device before resize2fs can grow the FS.  This is
    destructive to the partition table -- gated behind --yes.
    """
    partition = args.partition or f"{args.device}3"
    print(f"about to delete and recreate partition 3 on {args.device}, then resize {partition}",
          file=sys.stderr)
    if not args.yes:
        print("re-run with --yes to proceed", file=sys.stderr)
        return 2

    # Equivalent to:
    #   (echo d; echo 3; echo n; echo 3; echo ""; echo ""; echo y; echo w) | sudo fdisk DEVICE
    fdisk_script = "d\n3\nn\n3\n\n\ny\nw\n"
    print(f"+ sudo fdisk {args.device}  (scripted)", file=sys.stderr)
    rc = subprocess.run(
        ["sudo", "fdisk", args.device],
        input=fdisk_script, text=True, check=False,
    ).returncode
    # fdisk returns nonzero when re-reading the partition table fails on a
    # mounted device, which is expected here -- the kernel will pick up the
    # new size after partprobe.  Only bail on hard failures.
    if rc not in (0, 1):
        print(f"fdisk failed (rc={rc})", file=sys.stderr)
        return rc

    run(["sudo", "partprobe", args.device], check=False)
    return run(["sudo", "resize2fs", partition], check=False)


# ---------------------------------------------------------------------------
# Subcommand: vms create
# ---------------------------------------------------------------------------

def cmd_vms_create(args: argparse.Namespace) -> int:
    for name in args.names:
        rc = run(["sudo", "bash", str(ROOT / "create_vm.sh"), name], check=False)
        if rc != 0:
            print(f"create_vm.sh {name} failed (rc={rc})", file=sys.stderr)
            return rc
    return 0


# ---------------------------------------------------------------------------
# Subcommand: install
# ---------------------------------------------------------------------------

def cmd_install(args: argparse.Namespace) -> int:
    """Push install/ to /home/ubuntu/install and run install.sh STACK ... inside the VM."""
    ssh_opts = os.environ.get("SSH_OPTS",
                              "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR").split()
    # Find the VM IP.
    out = subprocess.check_output(
        ["sudo", "virsh", "domifaddr", args.vm], text=True)
    ip = None
    for line in out.splitlines():
        if "vnet" in line:
            ip = line.split()[3].split("/")[0]
            break
    if not ip:
        print(f"could not resolve IP for {args.vm}", file=sys.stderr)
        return 2

    # rsync install/ into the VM.
    rsync_cmd = ["rsync", "-az", "-e", "ssh " + " ".join(ssh_opts),
                 f"{INSTALL}/", f"ubuntu@{ip}:/home/ubuntu/install/"]
    run(rsync_cmd)

    # Run installer.
    ssh_cmd = ["ssh", *ssh_opts, f"ubuntu@{ip}",
               "bash /home/ubuntu/install/install.sh " + " ".join(args.stacks)]
    return run(ssh_cmd, check=False)


# ---------------------------------------------------------------------------
# Subcommand: run-mix
# ---------------------------------------------------------------------------

def cmd_run_mix(args: argparse.Namespace) -> int:
    return run_mix(
        workloads=args.workloads,
        vm_ids=args.vm_ids,
        wait_policy="all",
        sample_interval=args.interval,
        output_dir=args.out,
    )


# ---------------------------------------------------------------------------
# Subcommand: run-target
# ---------------------------------------------------------------------------

def cmd_run_target(args: argparse.Namespace) -> int:
    """Measure args.target while args.background runs in parallel.

    --neighbor-mode controls whether the background mix is real workloads or
    a fleetbench-driven synthetic mix.
    """
    registry = load_registry()
    if args.target not in registry:
        print(f"unknown target workload: {args.target}", file=sys.stderr)
        return 2

    bg_workloads = list(args.background or [])
    bg_vm_ids = [int(v) for v in (args.background_vms or [])]
    if len(bg_workloads) != len(bg_vm_ids):
        print("--background and --background-vms must match in length", file=sys.stderr)
        return 2

    # Auto-pick a VM for the target if not specified.
    target_vm = args.target_vm
    if target_vm is None:
        wtype = registry[args.target]["type"]
        pool = vm_pool_for_type(wtype)
        free = [v for v in pool if v not in bg_vm_ids]
        if not free:
            print(f"no free VM in pool for type {wtype}", file=sys.stderr)
            return 2
        target_vm = free[0]

    # In synthetic mode the background list is the "neighbors we replace": we
    # still need to know how many cores they would have taken so the target
    # gets pinned to the right offset.
    if args.neighbor_mode == "synthetic":
        offset = sum(read_num_cores(w) for w in bg_workloads)
        workloads = [args.target]
        vm_ids = [target_vm]
        target_idx = 0
        core_base = offset
    else:
        workloads = bg_workloads + [args.target]
        vm_ids = bg_vm_ids + [target_vm]
        target_idx = len(workloads) - 1
        core_base = 0

    extra_env = {}
    lock_file = args.lock_file
    if args.neighbor_mode == "synthetic":
        if not args.fleetbench:
            print("--fleetbench is required for --neighbor-mode synthetic", file=sys.stderr)
            return 2
        if not lock_file:
            lock_file = f"/tmp/mimebench_synt_{args.target}.lock"
        extra_env["FLEETBENCH_PATH"] = args.fleetbench
        extra_env["SYNT_LOCK_PATH"] = lock_file
        if args.synth_plan:
            extra_env["SYNT_H5_PATH"] = args.synth_plan

    return run_mix(
        workloads=workloads,
        vm_ids=vm_ids,
        wait_policy="target",
        target_idx=target_idx,
        core_base=core_base,
        neighbor_mode=args.neighbor_mode,
        lock_file=lock_file,
        sample_interval=args.interval,
        output_dir=args.out,
        extra_env=extra_env,
    )


# ---------------------------------------------------------------------------
# Subcommand: run-batch (replaces generate_noisy_neighbor_exps*.py)
# ---------------------------------------------------------------------------

def load_config_dir(config_dir: Path) -> dict[int, list[dict]]:
    """Parse exp_configs/<node>/<N>/config.txt as the old Python drivers did."""
    out: dict[int, list[dict]] = defaultdict(list)
    for n in range(1, 7):
        path = config_dir / str(n) / "config.txt"
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            workloads_part, vm_part = raw.split('" "')
            workloads = workloads_part.strip('"').split()
            vm_ids = [int(x) for x in vm_part.strip('"').split()]
            out[n].append({"workloads": workloads, "vm_ids": vm_ids})
    return out


def cmd_run_batch(args: argparse.Namespace) -> int:
    registry = load_registry()
    target = args.target
    if target not in registry:
        print(f"unknown target: {target}", file=sys.stderr)
        return 2

    cfgs = load_config_dir(Path(args.config_dir))
    target_type = registry[target]["type"]
    target_pool = vm_pool_for_type(target_type)
    max_cpu = int(os.environ.get("MAX_CPU_CORES", "20"))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    flat = []
    for num_vms, mixes in cfgs.items():
        for idx, mix in enumerate(mixes):
            free = [v for v in target_pool if v not in mix["vm_ids"]]
            if not free:
                continue
            target_vm = free[0]

            cores_needed = sum(read_num_cores(w) for w in mix["workloads"] + [target])
            if cores_needed > max_cpu:
                print(f"skip mix {idx} (cores={cores_needed} > {max_cpu})", file=sys.stderr)
                continue

            flat.append((num_vms, idx, mix, target_vm))

    if args.limit:
        flat = flat[: args.limit]

    for num_vms, idx, mix, target_vm in flat:
        out_dir = out_root / target / f"{num_vms}_{idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        bg_workloads = mix["workloads"]
        bg_vm_ids = mix["vm_ids"]

        ns = argparse.Namespace(
            target=target,
            target_vm=target_vm,
            background=bg_workloads,
            background_vms=bg_vm_ids,
            neighbor_mode=args.neighbor_mode,
            lock_file=None,
            fleetbench=args.fleetbench,
            synth_plan=args.synth_plan_for(num_vms, idx) if callable(args.synth_plan_for) else None,
            interval=args.interval,
            out=str(out_dir),
        )
        rc = cmd_run_target(ns)
        if rc != 0:
            print(f"run-target failed (rc={rc}) for {num_vms}/{idx}", file=sys.stderr)
            if not args.keep_going:
                return rc
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mimebench")
    sub = p.add_subparsers(dest="cmd", required=True)

    # init (optional one-shot host setup steps)
    pin = sub.add_parser("init", help="one-shot host setup steps")
    pin_sub = pin.add_subparsers(dest="init_cmd", required=True)
    pir = pin_sub.add_parser(
        "resize-disk",
        help="(cloudlab) delete + recreate root partition to span the disk, then resize2fs")
    pir.add_argument("--device", default="/dev/sda",
                     help="block device to repartition (default /dev/sda)")
    pir.add_argument("--partition", default=None,
                     help="partition to resize2fs (default: <device>3)")
    pir.add_argument("--yes", action="store_true",
                     help="confirm the destructive partition rewrite")
    pir.set_defaults(func=cmd_init_resize_disk)

    # vms create
    pv = sub.add_parser("vms", help="VM lifecycle")
    pv_sub = pv.add_subparsers(dest="vm_cmd", required=True)
    pvc = pv_sub.add_parser("create", help="create libvirt VMs via create_vm.sh")
    pvc.add_argument("names", nargs="+", help="vm1 vm2 ...")
    pvc.set_defaults(func=cmd_vms_create)

    # install
    pi = sub.add_parser("install", help="rsync install/ into a VM and run install.sh STACK ...")
    pi.add_argument("--vm", required=True, help="libvirt domain name (e.g. vm1)")
    pi.add_argument("stacks", nargs="+",
                    help="stack name(s) under install/: database, bigdata, kvstore, web, ml, graph, deathstarbench, dlrm, tpch, spec")
    pi.set_defaults(func=cmd_install)

    # run-mix
    pm = sub.add_parser("run-mix", help="run N workloads in N VMs in parallel")
    pm.add_argument("--workloads", required=True, nargs="+")
    pm.add_argument("--vm-ids", required=True, nargs="+", type=int)
    pm.add_argument("--interval", type=int, default=1, help="sampler interval, seconds")
    pm.add_argument("--out", default=None, help="output directory")
    pm.set_defaults(func=cmd_run_mix)

    # run-target
    pt = sub.add_parser("run-target",
                        help="measure target workload while a background mix co-runs")
    pt.add_argument("--target", required=True)
    pt.add_argument("--target-vm", type=int, default=None,
                    help="override VM for target (default: first free VM from the type pool)")
    pt.add_argument("--background", nargs="*", default=[],
                    help="co-running workloads (real ones in --neighbor-mode real, "
                         "or the mix being *replaced* in --neighbor-mode synthetic)")
    pt.add_argument("--background-vms", nargs="*", default=[],
                    help="VM IDs for --background (must match in length)")
    pt.add_argument("--neighbor-mode", choices=("none", "real", "synthetic"), default="real")
    pt.add_argument("--lock-file", default=None)
    pt.add_argument("--fleetbench", default=os.environ.get("FLEETBENCH_PATH"),
                    help="required when --neighbor-mode synthetic")
    pt.add_argument("--synth-plan", default=None,
                    help="path to fleetbench .h5 plan to install")
    pt.add_argument("--interval", type=int, default=1)
    pt.add_argument("--out", default=None)
    pt.set_defaults(func=cmd_run_target)

    # run-batch
    pb = sub.add_parser("run-batch",
                        help="iterate over exp_configs/<node>/<N>/config.txt and run-target each")
    pb.add_argument("--target", required=True)
    pb.add_argument("--config-dir", default="exp_configs/node12")
    pb.add_argument("--out", default="data/benchmarks")
    pb.add_argument("--neighbor-mode", choices=("none", "real", "synthetic"), default="real")
    pb.add_argument("--fleetbench", default=os.environ.get("FLEETBENCH_PATH"))
    pb.add_argument("--synth-plan-dir", default=None,
                    help="directory of plan_num_vms_<N>_<idx>_*.h5 files (synthetic mode)")
    pb.add_argument("--interval", type=int, default=1)
    pb.add_argument("--limit", type=int, default=None)
    pb.add_argument("--keep-going", action="store_true")
    pb.set_defaults(func=cmd_run_batch)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # For run-batch, attach a synth_plan_for(num_vms, idx) resolver.
    if getattr(args, "cmd", None) == "run-batch":
        plan_dir = Path(args.synth_plan_dir) if args.synth_plan_dir else None

        def _resolve(num_vms: int, idx: int):
            if plan_dir is None:
                return None
            for root, _, files in os.walk(plan_dir):
                for f in sorted(files):
                    if not f.endswith(".h5"):
                        continue
                    # Filenames look like: plan_num_vms_<N>_<idx>_node12.h5
                    parts = f.split("plan_num_vms_", 1)
                    if len(parts) != 2:
                        continue
                    tail = parts[1].split("_")
                    if len(tail) < 2:
                        continue
                    try:
                        n_in_file = int(tail[0])
                        idx_in_file = int(tail[1])
                    except ValueError:
                        continue
                    if n_in_file == num_vms and idx_in_file == idx:
                        return str(Path(root) / f)
            return None
        args.synth_plan_for = _resolve

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
