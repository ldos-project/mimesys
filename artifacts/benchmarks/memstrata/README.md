# memstrata: workload-mix experiments

This directory runs mixed cloud workloads inside libvirt VMs, samples
host-level performance counters and energy every N seconds, and supports
both real and synthetic co-runners (fleetbench).

The original upstream memstrata kernel/QEMU/orchestrator pieces are
preserved (`build_and_install_kernel.sh`, `build_and_install_qemu.sh`,
`build_orchestrator.sh`, `orchestrator/`); the per-VM NUMA-binding option
that came with them is no longer used.

## Layout

```
config/
  host.env             # paths, sample interval, VM pools per type
  workloads.tsv        # workload registry (name, type, install_stack)
install/
  install.sh           # dispatcher: install.sh STACK [STACK ...]
  database.sh bigdata.sh kvstore.sh web.sh deathstarbench.sh
  ml.sh graph.sh dlrm.sh tpch.sh spec.sh
lib/
  run_mix.sh           # unified mixed-workload runner (host side)
  vm_xml.sh ssh_helpers.sh samplers.sh
  neighbor/
    none.sh real.sh synthetic.sh
mimebench.py           # Python CLI on top of run_mix.sh
workload_scripts/<name>/{exp_config.sh, prepare_exp.sh, run_exp.sh}
exp_configs/           # pre-computed workload mixes (used by run-batch)
legacy/                # the original bash + python drivers, kept for reference
```

## Five things you can do

### 0a. (CloudLab only) Resize the root disk
CloudLab images ship with a small root partition.  Grow partition 3 to fill
the device and then `resize2fs`:
```
./mimebench.py init resize-disk --yes              # defaults to /dev/sda / /dev/sda3
./mimebench.py init resize-disk --device /dev/nvme0n1 --partition /dev/nvme0n1p3 --yes
```
This is destructive to the partition table; `--yes` is required.  Skip this
step on any host where the root FS already spans the disk.

### 0b. Create the virtiofs share directory
Every VM mounts `$SHARED_DIR` (default `/dev/shm/shared`) as
`/home/ubuntu/shared`.  Create it once on each host:
```
./mimebench.py init shared-dir
```
`vms create` calls this implicitly, so you only need it explicitly if you
want to populate the share before any VMs exist.

### 1. Create VMs
```
./mimebench.py vms create vm1 vm2 vm3 vm4 vm5 vm6
```
Wraps `create_vm.sh`. After this, each VM has a virtiofs share mounted at
`/home/ubuntu/shared` (the host's `/dev/shm/shared`).

### 2. Install workloads + dependencies inside a VM
```
./mimebench.py install --vm vm6 database          # silo for TPC-C
./mimebench.py install --vm vm4 bigdata           # Hadoop + Spark + HiBench
./mimebench.py install --vm vm7 kvstore           # FASTER + Redis + Memcached + YCSB
./mimebench.py install --vm vm1 web               # DaCapo + Renaissance
./mimebench.py install --vm vm3 deathstarbench    # DSB social + media
./mimebench.py install --vm vm9 ml                # MLPerf (sdxl, resnet50)
./mimebench.py install --vm vm10 graph            # GAPBS

# Or, every real stack in one VM (long: hours):
./mimebench.py install --vm vm1 --all
```
SPEC, DLRM, and TPC-H require manual install steps; their stubs explain what
to do (`install/spec.sh`, `install/dlrm.sh`, `install/tpch.sh`).  `--all`
skips them — pass them explicitly alongside `--all` if you really want.

Different VMs can be installed in parallel safely (different rootfs, different
apt lock).  Same VM, multiple stacks: sequence them.

Pass a CSV to `--vm` to fan out one detached child per VM, each writing to
`/tmp/mimebench_logs/<vm>.log`:
```
./mimebench.py install --vm vm2,vm3,vm4,vm5,vm6,vm7,vm8,vm9 --all
# returns immediately; tail -f /tmp/mimebench_logs/vm*.log to follow
```
Add `--wait` to block until every child finishes.  Each child uses
`start_new_session=True` (Python equivalent of `setsid`) so installs survive
the parent shell exiting.

### 3. Run a workload mix and profile every N seconds
```
./mimebench.py run-mix \
    --workloads spark_terasort dacapo_tomcat memcached_ycsb_a \
    --vm-ids   4              1              8 \
    --interval 1 \
    --out data/mix-001
```
Runs all three in parallel, waits for *all* of them to finish, samples
`hpcperfstatsd` + powercap energy every second, and dumps:
```
data/mix-001/
  stats.txt
  power.log
  result_app_perf_4_spark_terasort.txt
  result_app_perf_1_dacapo_tomcat.txt
  result_app_perf_8_memcached_ycsb_a.txt
```

### 4. Measure one target app while a noisy-neighbor mix runs alongside
```
./mimebench.py run-target \
    --target silo_tpcc \
    --background dacapo_tomcat memcached_ycsb_a spark_terasort \
    --background-vms 1            8                 4 \
    --neighbor-mode real \
    --interval 1 \
    --out data/silo_tpcc-mix-001
```
The target VM is picked from the database VM pool (`VM_POOL_DATABASE` in
`config/host.env`).  The runner waits only for the *target's* `run_exp.sh`
to finish and then `pkill`s the neighbors.

### 5. Swap real co-runners for synthetic ones
```
./mimebench.py run-target \
    --target silo_tpcc \
    --background dacapo_tomcat memcached_ycsb_a spark_terasort \
    --background-vms 1            8                 4 \
    --neighbor-mode synthetic \
    --fleetbench /users/dhkim/fleetbench \
    --synth-plan /users/dhkim/synth_plans/plan_num_vms_3_0_node12.h5 \
    --interval 1 \
    --out data/silo_tpcc-synth-001
```
In `--neighbor-mode synthetic` the `--background` list is treated as the
mix being *replaced*: it is used only to compute how many CPU cores the
target should pin past.  Fleetbench launches its stressors via the
`SYNT_LOCK_PATH` 0→1→2 handshake.

### Batch driver (replaces `generate_noisy_neighbor_exps*.py`)
```
./mimebench.py run-batch \
    --target silo_tpcc \
    --config-dir exp_configs/node12 \
    --neighbor-mode real \
    --interval 1 \
    --out data/benchmarks

./mimebench.py run-batch \
    --target silo_tpcc \
    --config-dir exp_configs/node12 \
    --neighbor-mode synthetic \
    --fleetbench /users/dhkim/fleetbench \
    --synth-plan-dir /users/dhkim/synthetic_workloads/workload_mix/memcpy \
    --out data/benchmarks_synt_memcpy
```

## Configuration knobs

Edit `config/host.env` to change:
- `TACC_STATS_DIR`, `TACC_STATS_LOG_PATH` — where `hpcperfstatsd` lives / writes
- `SHARED_DIR` — virtiofs share (`/dev/shm/shared`)
- `SAMPLE_INTERVAL_SEC` — default sampler interval
- `VM_POOL_<TYPE>` — which VM IDs serve each workload type
- `MAX_CPU_CORES` — total vCPU budget across all co-running VMs

Workloads are classified in `config/workloads.tsv` (one line per workload:
`name<TAB>type<TAB>install_stack`).  Add a new workload by creating
`workload_scripts/<name>/` and appending one row.

## Building memstrata kernel / QEMU / orchestrator

Unchanged from upstream:
```
./build_and_install_kernel.sh
sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 5.19.0-memstrata+"
sudo reboot
./build_and_install_qemu.sh
./build_orchestrator.sh
```
An Intel Xeon 6 processor is required to reproduce the original paper.
