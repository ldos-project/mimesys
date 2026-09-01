# Pretrained checkpoints — union_v2 (2026-07)

DiT diffusion denoisers pretrained for 300 epochs (61 collection rounds, 17,448 samples, aug ×20).
Given a 28-metric hardware trace, the model generates a mimesys
execution plan (20-stressor × 20-thread weight grid per 1 s slot) that
reproduces the trace's resource footprint on the target machine.

| dir | conditioning | notes |
|---|---|---|
| `prev_none/`  | metrics only | baseline |
| `prev_token/` | metrics + per-token prev-action bias | ~10% better generation quality (CPU trace, TPC-C tput); requires `MIMESYS_DIT_PREV_COND=token` at load time |

Both are `model_arch=dit` (`mimesys/models/dit_cond.py`): a ViT-style diffusion transformer that patchifies the 20×20 action grid, with adaLN conditioning on the encoded
metrics + diffusion timestep, and learned stressor/thread position embeddings.

## Hardware environment

Trained on, and calibrated for, **CloudLab Wisconsin `c220g5`**. Specs as encoded in `mimesys/schema/constants.py`:

| component | value |
|---|---|
| CPU | 2 × Intel Xeon Silver 4114 (Skylake-SP), 10 cores/socket, 2.2 GHz base / 3.0 GHz turbo |
| Cores seen by mimesys | 20 physical cores, CPUs 0–19 (cores 0–9 = NUMA node 0, 10–19 = node 1) |
| Caches | 32 KB L1i + 32 KB L1d and 1 MB L2 per core; 13.75 MB L3 per socket (27.5 MB/node, non-inclusive) |
| Memory | 192 GB DDR4-2400 (6 × 32 GB, 3 channels/socket), ~115 GB/s peak BW |
| Disk (IO stressors) | Intel DC S3520 480 GB SATA SSD — measured ceilings ≈ 330 MB/s sustained reads, writes burst to ≈ 380 MB/s through the page cache |
| Network | 10 Gbps (unused by the model) |

The checkpoint is machine-specific in two ways: (1) input normalization
ranges (below) come from this box, and (2) the stressor kernels are
sized to its caches (e.g. L1/L2/LLC working sets of 32 KB / 512 KB /
12 MB, 32 MB "larger-than-LLC" streams) and to its SSD's throughput
curve. On different hardware the plans still run, but the metric →
action mapping is no longer calibrated. Note some c220g5 hosts enumerate
`sda`=HDD / `sdb`=SSD — check which device backs the stressor scratch
dir before trusting IO numbers.

pqos features are collected in-process via libpqos (Intel RDT, MSR
interface) with one merged monitoring group over CPUs 0–19.

## Serving

```bash
# prev_none
uv run python -m mimesys.inference \
    --ckpt artifacts/pretrained_union_v2/prev_none/diffusion-epoch=299.ckpt

# prev_token (env var selects the matching architecture — required)
MIMESYS_DIT_PREV_COND=token uv run python -m mimesys.inference \
    --ckpt artifacts/pretrained_union_v2/prev_token/diffusion-epoch=299.ckpt
```

Both use the default `--exp pretrain` config. Add `--enable_profiling`
for `POST /profile` (needs SSH access to the worker fleet in the exp
config). Then e.g.:

```bash
uv run python -m mimesys.inference.client generate-from-file \
    --file /path/to/stats-workload.txt --method diffusion --output plan.h5
```

## Model inputs (28-dim, per 1 s time step)

One 28-dim vector per second of trace, from an HPCPerfStats stats file +
paired pqos log (`stats-<name>.txt` / `pqos-<name>.log`; the client
merges them automatically):

| # | metric | unit | training range |
|---|---|---|---|
| 0–19 | `avg_cpu_utilizations_core_00..19` | % per physical core | 0–100 |
| 20 | `io_read` | KB/s | 0–488,797 |
| 21 | `io_write` | KB/s | 0–466,290 |
| 22 | `l3_cache_usage` | MB | 0–3,379 |
| 23 | `memory_bandwidth_read` | GB/s | 0–80.6 |
| 24 | `memory_bandwidth_write` | GB/s | 0–40.8 |
| 25 | `pqos_llc_kb` | KB (RDT LLC occupancy) | 0–28,160 |
| 26 | `pqos_ipc` | instr/cycle | 0–3.5 |
| 27 | `pqos_misses` | LLC misses/s | 0–1.503e9 |

Inputs are min–max normalized to [-1, 1] with these ranges (clipped).
Values outside the training range degrade quality.

## Model outputs (execution plan)

Per 1 s slot, a **20 × 20 grid**: rows = the 20 stressors below, columns
= worker threads 0–19 (thread *i* pinned to CPU *i*). Each cell is a
weight in [0, 1]:

- **Weight = duty cycle.** The runner alternates busy/sleep in 5 ms
  chunks so a thread's CPU% tracks its weight near-linearly
  (`RunWithDutyCycle` in `mimesys_benchmark.cc`).
- **Per-thread column sum is capped at 1.0** during collection (Dirichlet
  budget cap), so training labels never encode an oversubscribed thread.
- For IO stressors the weight additionally scales the **bytes per IO
  call** (`target = weight × block_size`), so IO throughput has a
  monotone dose-response instead of saturating at the device ceiling.

Stressor-name conventions: `NoSleep` = tight loop, no throttle;
`WeightScaled_Xms` = `usleep(X)` between IO calls (larger X ⇒ lower
throughput floor; 500 ms variants give the kernel time to act on
`posix_fadvise(DONTNEED)` so reads hit disk, covering the sub-10 MB/s
range); `BurstScaled` = tight loop where weight scales only the
per-write size, restoring dose-response for writes that would otherwise
saturate writeback.

## The 20 actions

Rows in order (also `fleetbench/mimesys/mimesys_actions.txt`). Kernels
0–7 and 15–19 are CPU/cache/memory-bound; 8–14 are disk IO.

| # | action | what it does | primary pressure |
|---|---|---|---|
| 0 | `BM_HASHING_Computecrc32c_Fleet_cold` | Tight loop of `absl::ComputeCrc32c` over cache-cold buffers with Fleet-distributed sizes | CPU (integer/crc pipe), cache misses |
| 1 | `BM_LIBC_Memcpy_Fleet_L1` | `memcpy` of Fleet-distributed small sizes (16 B–2 KB) inside a 32 KB buffer (L1d-resident) | CPU, L1 |
| 2 | `BM_LIBC_Memcpy_Fleet_L2` | Same, sizes 1–64 KB inside a 512 KB buffer (half of the 1 MB L2) | CPU, L2 |
| 3 | `BM_LIBC_Memcmp_Fleet_LLC` | `memcmp` over two 12 MB buffers (~half of one socket's LLC), NUMA-local, large chunks | LLC occupancy + hit BW |
| 4 | `BM_SIMD_SerialDistanceComputation` (256 blocks, no AVX-512) | Vectorized (AVX2) serial distance computation over 256 data blocks | CPU FP/SIMD ports |
| 5 | `BM_SWISSMAP_InsertMiss_Cold flat_hash_set 262144` | Repeatedly fills and drops a fresh 262 k-entry `absl::flat_hash_set` with always-miss inserts | CPU + allocation + cache misses |
| 6 | `BM_SWISSMAP_InsertMiss_Cold node_hash_set 32768` | Same with a 32 k-entry `node_hash_set` (per-node allocations ⇒ pointer-chasing) | CPU + allocator |
| 7 | `BM_SWISSMAP_InsertManyOrdered_Cold flat_hash_set 1048576` | Ordered bulk insert of 1 M entries into a fresh `flat_hash_set` (density 1) | CPU + LLC + DRAM (rehash streams) |
| 8 | `BM_STRESS_NG_HddRead_1MB_NoSleep` | Untamed read loop: `posix_fadvise(DONTNEED)` + `pread` of weight×1 MB from a shared file, per-thread offsets | Disk read (saturates ~330 MB/s ceiling) |
| 9 | `BM_STRESS_NG_HddRead_1MB_WeightScaled_50ms` | Same read, 50 ms sleep per iteration | Disk read, mid-range |
| 10 | `BM_STRESS_NG_HddRead_256KB_WeightScaled_500ms` | 256 KB reads, 500 ms sleep — DONTNEED completes between reads so they truly hit disk | Disk read, low-range (<10 MB/s) |
| 11 | `BM_STRESS_NG_HddWriteNF_1MB_BurstScaled` | Tight `pwrite` loop of weight×1 MB to a rolling unlinked file, `sync_file_range` async writeback, no fsync; weight scales burst size only | Disk write (burst/page-cache regime) |
| 12 | `BM_STRESS_NG_HddWriteNF_1MB_WeightScaled_50ms` | Same write path, 50 ms sleep per iteration | Disk write, mid-range |
| 13 | `BM_STRESS_NG_HddWriteNF_1MB_WeightScaled_25ms` | Same, 25 ms sleep (≈2× the 50 ms rate) | Disk write, mid-high |
| 14 | `BM_STRESS_NG_HddWriteNF_256KB_WeightScaled_500ms` | 256 KB writes, 500 ms sleep | Disk write, low-range |
| 15 | `BM_DIRECTMEMSET_32MB_NoSleep` | Repeated `memset` of a per-thread 32 MB buffer (exceeds LLC ⇒ constant eviction + writeback) | DRAM write BW |
| 16 | `BM_DIRECTMEMSET_32MB_WeightScaled_50ms` | Same, 50 ms sleep between passes | DRAM write BW, mid-range |
| 17 | `BM_DIRECTSTREAMSIMD_32MB_NoSleep` | AVX2 streaming read+write over a 32 MB buffer | DRAM read+write BW (max) |
| 18 | `BM_DIRECTSTREAMSIMD_8MB_NoSleep` | Same over 8 MB (partially LLC-resident) | LLC + DRAM BW blend |
| 19 | `BM_HASHING_Extendcrc32c_Fleet_L3_16MB` | `absl::ExtendCrc32c` streaming over a 16 MB working set (larger than one socket's L3) | CPU + LLC miss traffic |

All memory-heavy kernels first-touch their buffers NUMA-locally
(cores 0–9 → node 0, 10–19 → node 1), so cross-socket traffic is not
part of the generated signal.
