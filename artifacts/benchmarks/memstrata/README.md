# Managing Memory Tiers with CXL in Virtualized Environments

This repository contains source code and instructions to reproduce key results in the paper. There are three major components:

* Modified Linux kernel (based on v5.19) that supports page coloring and page exchange
* Modified QEMU (based on v6.2) with page coloring support
* Orchestrator for estimating VM slowdown and exchanging pages between VMs to improve performance




Intel® Xeon® 6 Processor is required to reproduce the results in the paper. We are working on generalizing the scripts so that people can run them even without Intel® Xeon® 6 Processor.

## Getting Started


First, clone this repository in a place that is large enough to compile Linux kernel:
```
git clone https://yuhong_zhong@bitbucket.org/yuhong_zhong/memstrata.git
cd memstrata
git submodule init
git submodule update --recursive
```


Compile and install Linux kernel:
```
./build_and_install_kernel.sh
```
This step will take some time since it needs to download the source code, install dependencies, and compile the kernel from scratch.


After the kernel is compiled and installed, you will be prompted to reboot into the Memstrata kernel:
```
sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 5.19.0-memstrata+"
sudo reboot
```
Note that other components can only be compiled when you are in the Memstrata kernel.


Then, build and install QEMU:
```
./build_and_install_qemu.sh
```


After that, build the orchestrator:
```
./build_orchestrator.sh
```


After building and installing all the components, you can create VMs by `./create_vm.sh [vm domain name (e.g., vm1)]`. In the scripts, we assume that the VMs are named as `vm1`, `vm2`, `vm3`, `vm4`, `vm5`, and `vm6`.


The workloads used in our evaluation can be installed by running `./vm_install_workloads.sh` inside each VM. The installation script should work in a clean Ubuntu 20.04 8-core 32GB VM with username `ubuntu` (VMs created using `./create_vm.sh` already satisfy these requirements). Note that SPEC and GAPBS workloads need to be installed manually inside the VM using the instructions from [Pond](https://github.com/MoatLab/Pond/tree/master/cpu2017).


The VM configuration (number of cores and VM memory size) of each workload can be found in the `workload_scripts` folder, along with the scripts to prepare and run each workload in a VM. 


To measure application performance using either local memory or CXL memory, we provide a script to start multiple VMs with given workloads and configure from which host NUMA node the VMs will allocate memory. For example, if we want to run `602.gcc_s` and `dlrm_rm1_high` inside two VMs in parallel using memory from NUMA 0, we can run `sudo bash ./run_exp_numa.sh "./workload_scripts/602.gcc_s ./workload_scripts/dlrm_rm1_high" 0`. The script will run the two given workloads in `vm1` and `vm2`, wait for them to finish, and store the results in the folder of the first workload (e.g., `./workload_scripts/602.gcc_s` in this case). The script will read the configuration of each workload, including the number of cores and memory size, and configure the corresponding VM. It will also take care of copying the scripts to run each workload into the VM and run them automatically.
