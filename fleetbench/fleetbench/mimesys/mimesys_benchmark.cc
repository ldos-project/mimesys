// Copyright 2022 The Fleetbench Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <thread>
#include <papi.h>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fcntl.h>
#include "absl/crc/crc32c.h"
#include <unistd.h>
#include <sys/ioctl.h>
#include <numaif.h>
#include <sched.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <fstream>


#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/base/no_destructor.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "fleetbench/common/common.h"
#include "fleetbench/dynamic_registrar.h"

// Compression
# include "fleetbench/mimesys/compression/compression_benchmark.h"

// Hashing
#include "fleetbench/mimesys/hashing/hashing_benchmark.h"

// Libc
#include "fleetbench/mimesys/libc/mem_benchmark.h"

// proto
#include "fleetbench/mimesys/proto/benchmark.h"

// simd
#include "fleetbench/mimesys/simd/simd_benchmark.h"
// stl
#include "fleetbench/mimesys/stl/cord_benchmark.h"
// swissmap
#include "fleetbench/mimesys/swissmap/swissmap_benchmark.h"

#include "fleetbench/mimesys/stress_ng/stress_ng_benchmark.h"

// All benchmarks in this file are for cold lookups.
namespace fleetbench {

namespace mimesys {

using ::benchmark::DoNotOptimize;

class SilentReporter : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
    void Finalize() override {}
};

// Returns a sorted list of the files for the distributions whose filenames
// start with 'prefix'.
static std::vector<std::filesystem::path> GetDistributionFiles(
    absl::string_view prefix) {
  return GetMatchingFiles(GetFleetbenchRuntimePath("mimesys/execution_plans"),
                          prefix);
}

static std::filesystem::path GetMemstrataCommandFile(absl::string_view prefix) {
  const auto& files = GetMatchingFiles(
    GetFleetbenchRuntimePath("mimesys/memstrata_commands"),
    prefix
  );
  return files.empty() ? std::filesystem::path() : files.front();
}

int CollectTACCStats(std::string tacc_stats_dir) {
  // Start the profiling process
  // This function executes a bash command to start hpcperfstatsd and returns the collect_pid.
  // Returns the PID of the background collection process.
  std::string cmd = R"(
    sudo )" + tacc_stats_dir + R"(/hpcperfstatsd collect &
  )";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to start hpcperfstatsd collection process." << std::endl;
    return -1;
  }
  return 0;
}

int StartTACCStats() {
  // Start the profiling process
  // This function executes a bash command to start hpcperfstatsd and returns the collect_pid.
  // Returns the PID of the background collection process.
  const char* log_env_dir = std::getenv("TACC_STATS_LOG_DIR");
  std::string tacc_stats_log_dir = log_env_dir ? log_env_dir : "/var/log/hpcperfstats"; // fallback if env not set
  std::filesystem::path src = std::filesystem::path(tacc_stats_log_dir) / "current";
  std::error_code ec;
  std::filesystem::remove(src, ec);
  if (ec) {
    std::cerr << "Failed to remove source file: " << ec.message() << std::endl;
  }

  const char* env_dir = std::getenv("TACC_STATS_DIR");
  std::string tacc_stats_dir = env_dir ? env_dir : "/users/dhkim/HPCPerfStats/monitor/src"; // fallback if env not set
  std::string cmd = R"(sudo )" + tacc_stats_dir + R"(/hpcperfstatsd begin)";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to start hpcperfstatsd collection process." << std::endl;
    return -1;
  }
  pclose(pipe);
  return 0;
}

std::string StopProfilingTACCStats() {
  const char* env_dir = std::getenv("TACC_STATS_LOG_DIR");
  std::string tacc_stats_dir = env_dir ? env_dir : "/var/log/hpcperfstats"; // fallback if env not set
  return tacc_stats_dir + "/current";
}


int StartProfilingTACCStats(int period) {
  // Start the profiling process
  // This function executes a bash command to start hpcperfstatsd and returns the collect_pid.
  // Returns the PID of the background collection process.
  const char* log_env_dir = std::getenv("TACC_STATS_LOG_DIR");
  std::string tacc_stats_log_dir = log_env_dir ? log_env_dir : "/var/log/hpcperfstats"; // fallback if env not set
  std::filesystem::path src = std::filesystem::path(tacc_stats_log_dir) / "current";
  std::error_code ec;
  std::filesystem::remove(src, ec);
  if (ec) {
    std::cerr << "Failed to remove source file: " << ec.message() << std::endl;
  }

  const char* env_dir = std::getenv("TACC_STATS_DIR");
  std::string tacc_stats_dir = env_dir ? env_dir : "/users/dhkim/HPCPerfStats/monitor/src"; // fallback if env not set
  std::string cmd = R"(
    sudo )" + tacc_stats_dir + R"(/hpcperfstatsd begin
    (
      while true; do
        sleep )" + std::to_string(static_cast<float>(period) / 10) + R"(
        sudo )" + tacc_stats_dir + R"(/hpcperfstatsd collect
      done
    ) &
    echo $!
  )";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to start hpcperfstatsd collection process." << std::endl;
    return -1;
  }
  char buffer[128];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
    break;
  }
  pclose(pipe);
  int collect_pid = std::stoi(result);
  return collect_pid;
}

int StartProfilingPCMThread(int period) {
  // Start the profiling process
  const char* env_dir = std::getenv("PCM_DIR");
  std::string pcm_path = env_dir ? env_dir : "/usr/sbin/pcm"; // fallback if env not set

  const char* log_env_dir = std::getenv("PCM_LOG_DIR");
  std::string pcm_log_dir = log_env_dir ? log_env_dir : "/users/dhkim/pcm_results/output.csv"; // fallback if env not set

  std::string cmd = R"(
    sudo )" + pcm_path + R"( )" + std::to_string(period) + R"( -csv=)" + pcm_log_dir + R"( &
    echo $!
  )";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to start pcm process." << std::endl;
    return -1;
  }
  char buffer[128];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
    break;
  }
  pclose(pipe);
  int collect_pid = std::stoi(result);

  std::cerr << "Sleeping for 30 seconds to allow profiler to start..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(30));
  return collect_pid;
}

int StartProfilingPCM() {
  const char* log_env_dir = std::getenv("PCM_LOG_DIR");
  std::string pcm_log_dir = log_env_dir ? log_env_dir : "/users/dhkim/pcm_results/output.csv"; // fallback if env not set

  std::string cmd = "sudo truncate -s 0 " + pcm_log_dir;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Failed to truncate PCM log file: " << pcm_log_dir << std::endl;
  }

  return 0;
}

std::string StopProfilingTACCStats(int collect_pid) {
  std::string cmd = "sudo kill " + std::to_string(collect_pid);
  system(cmd.c_str());

  const char* env_dir = std::getenv("TACC_STATS_LOG_DIR");
  std::string tacc_stats_dir = env_dir ? env_dir : "/var/log/hpcperfstats"; // fallback if env not set

  return tacc_stats_dir + "/current";
}

std::string StopProfilingPCM(int collect_pid) {
  // std::string cmd = "sudo kill " + std::to_string(collect_pid);
  // system(cmd.c_str());

  const char* log_env_dir = std::getenv("PCM_LOG_DIR");
  return log_env_dir ? log_env_dir : "/users/dhkim/pcm_results/output.csv"; // fallback if env not set
}

int StartProfiling() {
  int period = 1;
  // return StartProfilingPCM();
  // return StartProfilingTACCStats(period);
  return StartTACCStats();
}

int StartProfilingBackground(int period) {
  return StartProfilingPCMThread(period);
}

void StopProfiling(int collect_pid, std::string filename) {
  // Stop the profiling processes
  // std::string src_path = StopProfilingTACCStats(collect_pid);
  // std::string src_path = StopProfilingPCM(collect_pid);

  const char* env_dir = std::getenv("TACC_STATS_LOG_DIR");
  std::string tacc_stats_dir = env_dir ? env_dir : "/var/log/hpcperfstats"; // fallback if env not set
  std::string src_path = tacc_stats_dir + "/current";

  const char* results_env_dir = std::getenv("PROFILED_STATS_DIR");
  std::string target_dir = results_env_dir ? results_env_dir : "/users/dhkim/results"; // fallback if env not set

  std::filesystem::path src = std::filesystem::path(src_path);
  std::filesystem::path dst = std::filesystem::path(target_dir) / ("stats-" + filename + ".txt");

  std::filesystem::create_directories(dst.parent_path());
  std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);

  std::string cmd = "sudo truncate -s 0 " + src_path;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Failed to truncate PCM log file: " << src_path << std::endl;
  }
}

void StopProfilingBackground(int background_profiler_pid) {
  std::string cmd = "sudo kill " + std::to_string(background_profiler_pid);
  system(cmd.c_str());
}

// Maps the default benchmarks to their minimum iteration counts.
absl::NoDestructor<std::vector<std::pair<std::string, int>>> kDefaultActions({});

std::tuple<std::vector<int>, double> GetNumBenchmarkItersFromExecutionPlan(
  std::vector<double> execution_plan, long long time_budget_us) {
  std::vector<int> num_iters;
  // For each ratio, compute the number of iterations needed to reach the target instruction count.
  // The indices of execution_plan and kDefaultBenchmarks are the same.


  std::cerr << "[DEBUG] execution_plan: ";
  for (size_t i = 0; i < execution_plan.size(); ++i) {
    std::cerr << execution_plan[i] << " ";
  }

  double total_ratio = std::accumulate(execution_plan.begin(), execution_plan.end(), 0.0);
  double no_op_ratio = (1 - total_ratio) > 0 ? (1 - total_ratio) : 0.0;
  for (size_t i = 0; i < execution_plan.size(); ++i) {
    double ratio = execution_plan[i];

    if (total_ratio > 1) {
      // Normalize the ratio to ensure the total ratio is 1.
      ratio /= total_ratio;
    }

    double target_time = static_cast<double>(ratio * time_budget_us);

    // Get the instruction count per iteration from kDefaultBenchmarks.
    // The second item of kDefaultBenchmarks is the instruction count per iteration.
    auto it = kDefaultActions->begin();
    std::advance(it, i);
    int64_t time_per_iter = it->second;

    // Compute the number of iterations (find the closest integer)
    int num_iters_for_benchmark = 0;
    if (time_per_iter > 0) {
      double quotient = target_time / static_cast<double>(time_per_iter);
      num_iters_for_benchmark = static_cast<int>(std::round(quotient));
      if (num_iters_for_benchmark < 0) num_iters_for_benchmark = 0;
    }
    num_iters.push_back(num_iters_for_benchmark);
  }
  return std::make_tuple(num_iters, no_op_ratio);
}

std::vector<int> ToExecutionOrder(std::vector<int> num_iters) {
  // Each item in num_iters is the number of times the index will be executed.
  // Create a vector with indices where the number of indices should be the same as the value in num_iters.
  // e.g., num_iters={4,1} => output should be {0,0,0,0,1}
  std::vector<int> order;

  std::vector<size_t> indices(num_iters.size());
  std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., num_iters.size()-1
  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(indices.begin(), indices.end(), rng); // Shuffle the indices randomly

  std::cerr << "[DEBUG] num_iters: ";
  for (size_t idx : indices) {
    std::cerr << idx << ":" << num_iters[idx] << ", ";
    order.insert(order.end(), num_iters[idx], static_cast<int>(idx));
  }
  std::cerr << std::endl;

  return order;
}

// ── Direct-invoke kernels (skip GoogleBenchmark RunSpecifiedBenchmarks) ─────
//
// For a few hot CRC32 workloads, replace the framework round-trip with a
// direct tight loop calling absl::ExtendCrc32c / absl::ComputeCrc32c until the
// per-slot deadline expires. Buffer + str_lengths are built once per thread,
// reused across slot invocations.
namespace direct_kernels {

struct CrcCtx {
  std::string buffer;
  absl::string_view sv;
  std::vector<int> str_lengths;
  bool init_done = false;
};

static thread_local CrcCtx g_ctx;

static void InitCrcCtx() {
  if (g_ctx.init_done) return;
  // 64 MB buffer comfortably exceeds L3 on c220g5 (2*~25 MB).
  g_ctx.buffer.assign(64ull * 1024 * 1024, 'x');
  g_ctx.sv = absl::string_view(g_ctx.buffer);
  // Representative size distribution (bytes per call).
  static const int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
  g_ctx.str_lengths.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    g_ctx.str_lengths.push_back(sizes[i % 8]);
  }
  g_ctx.init_done = true;
}

static void DirectExtendCrc32c(long deadline_us) {
  InitCrcCtx();
  auto& ctx = g_ctx;
  absl::crc32c_t v0{0};
  size_t start = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - loop_start).count();
    if (elapsed_us >= deadline_us) break;
    for (auto l : ctx.str_lengths) {
      if (start + l >= ctx.sv.length()) start = 0;
      absl::string_view buf = ctx.sv.substr(start, l);
      start += l + 4096;
      v0 = absl::ExtendCrc32c(v0, buf);
      benchmark::DoNotOptimize(v0);
    }
  }
}

static void DirectComputeCrc32c(long deadline_us) {
  InitCrcCtx();
  auto& ctx = g_ctx;
  size_t start = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - loop_start).count();
    if (elapsed_us >= deadline_us) break;
    for (auto l : ctx.str_lengths) {
      if (start + l >= ctx.sv.length()) start = 0;
      absl::string_view buf = ctx.sv.substr(start, l);
      start += l + 4096;
      auto res = absl::ComputeCrc32c(buf);
      benchmark::DoNotOptimize(res);
    }
  }
}

struct MemcpyCtx {
  char* src = nullptr;
  char* dst = nullptr;
  size_t bufsize = 0;
  bool init = false;
};
static thread_local MemcpyCtx mem_ctx;

static void DirectMemcpy(long deadline_us) {
  if (!mem_ctx.init) {
    mem_ctx.bufsize = 32 * 1024;  // L1d size on c220g5
    if (posix_memalign(reinterpret_cast<void**>(&mem_ctx.src), 512, mem_ctx.bufsize) != 0
        || posix_memalign(reinterpret_cast<void**>(&mem_ctx.dst), 512, mem_ctx.bufsize) != 0) {
      return;  // alloc failed
    }
    std::memset(mem_ctx.src, 0xAB, mem_ctx.bufsize);
    std::memset(mem_ctx.dst, 0x00, mem_ctx.bufsize);
    mem_ctx.init = true;
  }
  // Distribution of copy sizes (roughly matches Fleet small-string memcpy).
  static constexpr int kSizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
  size_t src_off = 0, dst_off = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    for (int len : kSizes) {
      if (src_off + len > mem_ctx.bufsize) src_off = 0;
      if (dst_off + len > mem_ctx.bufsize) dst_off = 0;
      auto* r = std::memcpy(mem_ctx.dst + dst_off, mem_ctx.src + src_off, len);
      benchmark::DoNotOptimize(r);
      src_off += len;
      dst_off += len;
    }
  }
}

// Bind a freshly-allocated (still-unmapped) buffer to the local NUMA node of
// the calling thread. Must be called BEFORE first-touch (i.e., before any
// memset/memcpy into the buffer) so the physical pages are placed on the
// local node. c220g5: cores 0..9 → node 0, cores 10..19 → node 1.
static void BindToLocalNumaNode(void* addr, size_t len) {
  int cpu = sched_getcpu();
  int node = (cpu >= 10) ? 1 : 0;
  unsigned long mask = 1UL << node;
  // maxnode must be > largest valid node ID; mbind expects it as bits + 1.
  // We use 64 (size in bits of one unsigned long) which covers up to node 63.
  long rc = mbind(addr, len, MPOL_BIND, &mask, 8 * sizeof(mask), 0);
  (void)rc;  // best-effort; if mbind unavailable we silently fall back
}

// LLC-resident memory kernels: ~12 MB buffers per thread (~LLC/2 on c220g5).
// At single-thread, this fits in LLC and stays cache-resident; multi-thread
// aggregate working sets exceed LLC and start spilling to DRAM, producing
// the high-LLC + moderate-BW signatures observed in test workloads.
struct MemLlcCtx {
  char* a = nullptr;
  char* b = nullptr;
  size_t bufsize = 0;
  bool init = false;
};
static thread_local MemLlcCtx llc_mem_ctx;

static void EnsureLlcMemCtx() {
  if (llc_mem_ctx.init) return;
  llc_mem_ctx.bufsize = 12ull * 1024 * 1024;  // ~half LLC on c220g5 (24 MB total)
  if (posix_memalign(reinterpret_cast<void**>(&llc_mem_ctx.a), 4096, llc_mem_ctx.bufsize) != 0
      || posix_memalign(reinterpret_cast<void**>(&llc_mem_ctx.b), 4096, llc_mem_ctx.bufsize) != 0) {
    return;  // alloc failed
  }
  // Bind to local NUMA node BEFORE first-touch so pages physically land
  // on the same socket as this worker thread.
  BindToLocalNumaNode(llc_mem_ctx.a, llc_mem_ctx.bufsize);
  BindToLocalNumaNode(llc_mem_ctx.b, llc_mem_ctx.bufsize);
  std::memset(llc_mem_ctx.a, 0xAB, llc_mem_ctx.bufsize);
  // Set b equal to a so memcmp doesn't bail early on a mismatch.
  std::memset(llc_mem_ctx.b, 0xAB, llc_mem_ctx.bufsize);
  llc_mem_ctx.init = true;
}

static void DirectMemcmpLLC(long deadline_us) {
  EnsureLlcMemCtx();
  if (!llc_mem_ctx.init) return;
  // Larger chunk sizes so each call touches enough cache lines to drive
  // the LLC and so loop overhead doesn't dominate.
  static constexpr int kSizes[] = {4096, 16384, 65536, 262144, 1048576};
  size_t off_a = 0, off_b = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    for (int len : kSizes) {
      if (off_a + len > llc_mem_ctx.bufsize) off_a = 0;
      if (off_b + len > llc_mem_ctx.bufsize) off_b = 0;
      int r = std::memcmp(llc_mem_ctx.a + off_a, llc_mem_ctx.b + off_b, len);
      benchmark::DoNotOptimize(r);
      // Stride forward; offset b by an extra cacheline so the two streams
      // walk different parts of the buffer simultaneously.
      off_a += len;
      off_b += len + 64;
    }
  }
}

static void DirectMemmoveLLC(long deadline_us) {
  EnsureLlcMemCtx();
  if (!llc_mem_ctx.init) return;
  static constexpr int kSizes[] = {4096, 16384, 65536, 262144, 1048576};
  size_t off_src = 0, off_dst = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    for (int len : kSizes) {
      if (off_src + len > llc_mem_ctx.bufsize) off_src = 0;
      if (off_dst + len > llc_mem_ctx.bufsize) off_dst = 0;
      auto* r = std::memmove(llc_mem_ctx.b + off_dst, llc_mem_ctx.a + off_src, len);
      benchmark::DoNotOptimize(r);
      off_src += len;
      off_dst += len + 64;
    }
  }
}

// L2-resident memory kernels: ~512 KB buffers per thread (~L2/2 on c220g5;
// L2 is 1 MB/core on Skylake-X). Single-thread stays in L2; multi-thread
// aggregate working set fits in shared LLC without spilling to DRAM,
// producing the high-LLC + low-DRAM-BW signature characteristic of cache-
// warm test workloads.
struct MemL2Ctx {
  char* a = nullptr;
  char* b = nullptr;
  size_t bufsize = 0;
  bool init = false;
};
static thread_local MemL2Ctx l2_mem_ctx;

static void EnsureL2MemCtx() {
  if (l2_mem_ctx.init) return;
  l2_mem_ctx.bufsize = 512ull * 1024;  // half L2 on c220g5 (L2 = 1 MB/core)
  if (posix_memalign(reinterpret_cast<void**>(&l2_mem_ctx.a), 4096, l2_mem_ctx.bufsize) != 0
      || posix_memalign(reinterpret_cast<void**>(&l2_mem_ctx.b), 4096, l2_mem_ctx.bufsize) != 0) {
    return;
  }
  // Bind to local NUMA node BEFORE first-touch.
  BindToLocalNumaNode(l2_mem_ctx.a, l2_mem_ctx.bufsize);
  BindToLocalNumaNode(l2_mem_ctx.b, l2_mem_ctx.bufsize);
  std::memset(l2_mem_ctx.a, 0xCD, l2_mem_ctx.bufsize);
  std::memset(l2_mem_ctx.b, 0xCD, l2_mem_ctx.bufsize);
  l2_mem_ctx.init = true;
}

static void DirectMemcpyL2(long deadline_us) {
  EnsureL2MemCtx();
  if (!l2_mem_ctx.init) return;
  static constexpr int kSizes[] = {1024, 4096, 16384, 65536};
  size_t off_src = 0, off_dst = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    for (int len : kSizes) {
      if (off_src + len > l2_mem_ctx.bufsize) off_src = 0;
      if (off_dst + len > l2_mem_ctx.bufsize) off_dst = 0;
      auto* r = std::memcpy(l2_mem_ctx.b + off_dst, l2_mem_ctx.a + off_src, len);
      benchmark::DoNotOptimize(r);
      off_src += len;
      off_dst += len + 64;
    }
  }
}

static void DirectMemcmpL2(long deadline_us) {
  EnsureL2MemCtx();
  if (!l2_mem_ctx.init) return;
  static constexpr int kSizes[] = {1024, 4096, 16384, 65536};
  size_t off_a = 0, off_b = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    for (int len : kSizes) {
      if (off_a + len > l2_mem_ctx.bufsize) off_a = 0;
      if (off_b + len > l2_mem_ctx.bufsize) off_b = 0;
      int r = std::memcmp(l2_mem_ctx.a + off_a, l2_mem_ctx.b + off_b, len);
      benchmark::DoNotOptimize(r);
      off_a += len;
      off_b += len + 64;
    }
  }
}

// SIMD-friendly direct kernel: auto-vectorizes to AVX FMA.
static void DirectSerialDistance(long deadline_us) {
  static thread_local std::vector<float> a, b, c;
  if (a.empty()) {
    a.resize(1024); b.resize(1024); c.resize(1024, 0.0f);
    for (int i = 0; i < 1024; ++i) {
      a[i] = static_cast<float>(i) * 0.5f;
      b[i] = static_cast<float>(i) * 0.3f;
    }
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    for (int i = 0; i < 1024; ++i) c[i] += a[i] * b[i];
    benchmark::DoNotOptimize(c.data());
    // Reset to keep magnitudes bounded.
    if (c[0] > 1e30f) std::fill(c.begin(), c.end(), 0.0f);
  }
}

// SwissMap direct kernels: fill+drop a fresh hash set repeatedly.
// Checks the deadline every 1024 insertions (and breaks the inner fill loop
// early if exceeded) so the duty-cycle wrapper can throttle large-N sets
// without overshooting its busy budget by ~tens-of-ms.
template <size_t N>
static void DirectSwissmapInsert(long deadline_us, bool ordered) {
  constexpr size_t kCheckMask = 1023;  // check every 1024 insertions
  auto loop_start = std::chrono::steady_clock::now();
  auto deadline_passed = [&]() {
    auto e = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    return e >= deadline_us;
  };
  while (!deadline_passed()) {
    absl::flat_hash_set<uint64_t> s;
    s.reserve(N);
    if (ordered) {
      for (uint64_t i = 0; i < N; ++i) {
        s.insert(i);
        if ((i & kCheckMask) == kCheckMask && deadline_passed()) break;
      }
    } else {
      // "Miss" pattern: pseudo-random keys (mixed via golden-ratio hash).
      uint64_t k = 0x9E3779B97F4A7C15ULL;
      for (uint64_t i = 0; i < N; ++i) {
        k = k * 2654435761ULL + i;
        s.insert(k);
        if ((i & kCheckMask) == kCheckMask && deadline_passed()) break;
      }
    }
    benchmark::DoNotOptimize(s.size());
  }
}
static void DirectSwissmapInsertMiss_32K(long d)        { DirectSwissmapInsert<32768>(d, false); }
static void DirectSwissmapInsertMiss_262K(long d)       { DirectSwissmapInsert<262144>(d, false); }
static void DirectSwissmapInsertMiss_1M(long d)         { DirectSwissmapInsert<1048576>(d, false); }
static void DirectSwissmapInsertOrdered_262K(long d)    { DirectSwissmapInsert<262144>(d, true); }
static void DirectSwissmapInsertOrdered_1M(long d)      { DirectSwissmapInsert<1048576>(d, true); }

// ── IO direct kernels (stress_ng-equivalent patterns, simplified) ──────────
//
// Each thread uses its own temp file under /tmp keyed by pthread_self() so
// concurrent workers don't collide. State is thread_local.

static std::string PerThreadPath(const char* tag) {
  char p[128];
  snprintf(p, sizeof(p), "/tmp/mimesys_direct_%s_%lu", tag,
           static_cast<unsigned long>(pthread_self()));
  return std::string(p);
}

static void DirectReadahead(long deadline_us) {
  static thread_local int fd = -1;
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static constexpr size_t kFileBytes = 64 * 1024 * 1024;  // 64 MB
  static constexpr size_t kChunk     = 64 * 1024;         // 64 KB
  if (fd < 0) {
    path = PerThreadPath("readahead");
    fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    // Unlink from filesystem now; the kernel keeps the inode alive as long as
    // fd is open, but disk space is reclaimed immediately when the process
    // exits (even via SIGKILL). Prevents the 64 MB per-thread file from
    // leaking across benchmark invocations.
    unlink(path.c_str());
    buf.assign(kChunk, 'X');
    for (size_t i = 0; i < kFileBytes / kChunk; ++i) {
      if (write(fd, buf.data(), buf.size()) <= 0) break;
    }
    fsync(fd);
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    posix_fadvise(fd, 0, kFileBytes, POSIX_FADV_DONTNEED);
    posix_fadvise(fd, 0, kFileBytes, POSIX_FADV_WILLNEED);
    lseek(fd, 0, SEEK_SET);
    for (size_t i = 0; i < kFileBytes / kChunk; ++i) {
      auto n = read(fd, buf.data(), buf.size());
      benchmark::DoNotOptimize(n);
      if (n <= 0) break;
    }
  }
}

static void DirectFallocate4MB(long deadline_us) {
  static thread_local std::string path;
  static constexpr off_t kSize = 4 * 1024 * 1024;
  if (path.empty()) path = PerThreadPath("fallocate");
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) continue;
    int rc = fallocate(fd, 0, 0, kSize);
    benchmark::DoNotOptimize(rc);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
  }
}

// Small-file fallocate: 256 KB preallocation per iteration. Lower-IO variant
// covering the small-block low-bandwidth IO regime.
static void DirectFallocate256KB(long deadline_us) {
  static thread_local std::string path;
  static constexpr off_t kSize = 256 * 1024;
  if (path.empty()) path = PerThreadPath("fallocate256");
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) continue;
    int rc = fallocate(fd, 0, 0, kSize);
    benchmark::DoNotOptimize(rc);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
  }
}

static void DirectHdd1MB(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static constexpr size_t kSize = 1024 * 1024;
  if (buf.empty()) {
    buf.assign(kSize, 'Z');
    path = PerThreadPath("hdd");
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) continue;
    auto n = write(fd, buf.data(), buf.size());
    benchmark::DoNotOptimize(n);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
  }
}

// Low-IO 64 KB Hdd variant: ~16× smaller write per iteration than Hdd_1MB.
// Sustains lower aggregate IO bandwidth, covering the low-IO regime of test
// workloads that the 1 MB variant misses.
static void DirectHdd64KB(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static constexpr size_t kSize = 64 * 1024;
  if (buf.empty()) {
    buf.assign(kSize, 'Z');
    path = PerThreadPath("hdd64");
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) continue;
    auto n = write(fd, buf.data(), buf.size());
    benchmark::DoNotOptimize(n);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
  }
}

using KernelFn = std::function<void(long)>;
static const std::unordered_map<std::string, KernelFn>& Kernels() {
  static const std::unordered_map<std::string, KernelFn> m = {
    {"BM_HASHING_Extendcrc32cinternal_Fleet_cold", DirectExtendCrc32c},
    {"BM_HASHING_Computecrc32c_Fleet_cold",        DirectComputeCrc32c},
    {"BM_LIBC_Memcpy_Fleet_L1",                    DirectMemcpy},
    {"BM_LIBC_Memcmp_Fleet_LLC",                   DirectMemcmpLLC},
    {"BM_LIBC_Memmove_Fleet_LLC",                  DirectMemmoveLLC},
    {"BM_LIBC_Memcpy_Fleet_L2",                    DirectMemcpyL2},
    {"BM_LIBC_Memcmp_Fleet_L2",                    DirectMemcmpL2},
    // SIMD: both 256-block and 512-block variants share the same direct kernel
    // (representative AVX-FMA loop); the original SCANN LUT16 distance compute
    // is a similar pattern.
    {"BM_SIMD_SerialDistanceComputation/num_blocks:256/enable_avx512:false/flush_cache:false", DirectSerialDistance},
    {"BM_SIMD_SerialDistanceComputation/num_blocks:512/enable_avx512:false/flush_cache:false", DirectSerialDistance},
    // SwissMap: parameterize by set_size; node_hash_set and flat_hash_set use
    // the same kernel (absl::flat_hash_set) — exact data structure differs but
    // the CPU profile is dominated by the insertion+probing loop.
    {"BM_SWISSMAP_InsertMiss_Cold<::absl::flat_hash_set, 64>/set_size:262144/density:0",      DirectSwissmapInsertMiss_262K},
    {"BM_SWISSMAP_InsertMiss_Cold<::absl::flat_hash_set, 64>/set_size:1048576/density:0",     DirectSwissmapInsertMiss_1M},
    {"BM_SWISSMAP_InsertMiss_Cold<::absl::node_hash_set, 64>/set_size:32768/density:0",       DirectSwissmapInsertMiss_32K},
    {"BM_SWISSMAP_InsertManyOrdered_Cold<::absl::flat_hash_set, 64>/set_size:262144/density:1", DirectSwissmapInsertOrdered_262K},
    {"BM_SWISSMAP_InsertManyOrdered_Cold<::absl::flat_hash_set, 64>/set_size:1048576/density:1", DirectSwissmapInsertOrdered_1M},
    {"BM_SWISSMAP_InsertManyOrdered_Cold<::absl::node_hash_set, 64>/set_size:1048576/density:1", DirectSwissmapInsertOrdered_1M},
    // IO stressors — replicate stress-ng patterns via direct syscalls.
    // Per-thread temp file under /tmp keyed by pthread_self() to avoid
    // contention between concurrent worker threads.
    {"BM_STRESS_NG_Readahead",       DirectReadahead},
    {"BM_STRESS_NG_Fallocate_4MB",   DirectFallocate4MB},
    {"BM_STRESS_NG_Fallocate_256KB", DirectFallocate256KB},
    {"BM_STRESS_NG_Hdd_1MB",         DirectHdd1MB},
    {"BM_STRESS_NG_Hdd_64KB",        DirectHdd64KB},
  };
  return m;
}

// Sub-iteration duty cycling: alternate (chunk*duty) busy with
// (chunk*(1-duty)) sleep, repeating until deadline. The supplied KernelFn is
// called with progressively smaller deadlines so it can return inside one chunk.
// Used to make per-thread CPU% follow `duty` smoothly instead of saturating.
static void RunWithDutyCycle(const KernelFn& fn, long deadline_us, double duty) {
  if (deadline_us <= 0) return;
  if (duty >= 0.99) { fn(deadline_us); return; }
  if (duty <= 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  // 5 ms chunk gives ~5 % min duty (250 us busy floor) at decent granularity.
  // Light kernels (Memcpy, Crc32c, SIMD) honor the deadline at sub-µs and
  // deliver near-linear duty cycling. Heavy SwissMap_1M is left non-monotonic
  // here — its per-iteration allocation cost is the bottleneck and a larger
  // chunk doesn't help; would need a kernel-level refactor (persistent set).
  const long chunk_us = 5000;
  long busy_us = std::max<long>(250, static_cast<long>(chunk_us * duty));
  long sleep_us = std::max<long>(0, chunk_us - busy_us);
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto now = std::chrono::steady_clock::now();
    long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        now - loop_start).count();
    if (elapsed >= deadline_us) break;
    long remaining = deadline_us - elapsed;
    long this_busy = std::min(busy_us, remaining);
    fn(this_busy);
    long after_busy = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    long this_sleep = std::min<long>(sleep_us, deadline_us - after_busy);
    if (this_sleep > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(this_sleep));
    }
  }
}

}  // namespace direct_kernels

void RunMimesysExecutionOrder(
    const std::vector<int>& execution_order,
    double no_op_ratio,
    std::unordered_map<int, int>& index_invocation_count,
    SilentReporter& display_reporter,
    unsigned int duration_us = 1000000) {
    // int EventSet = PAPI_NULL;
    // // Create the EventSet
    // if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
    //   std::cerr << "PAPI create eventset error!" << std::endl;
    //   return;
    // }
    // // Add the total instructions event
    // if (PAPI_add_event(EventSet, PAPI_TOT_INS) != PAPI_OK) {
    //   std::cerr << "PAPI add event error!" << std::endl;
    //   return;
    // }
    // // Start counting instructions
    // if (PAPI_start(EventSet) != PAPI_OK) {
    //     std::cerr << "PAPI start counters error!" << std::endl;
    //     return;
    // }

    // long long values[1];
    // long long num_insts = 0;
    // long long prev_num_insts = 0;
    auto loop_start_time = std::chrono::steady_clock::now();
    auto loop_start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      loop_start_time.time_since_epoch()).count();

    auto time_diff_us = 0LL;
    auto prev_time_diff_us = 0LL;
    while (time_diff_us < duration_us) {
      for (int spec_idx : execution_order) {
        // Increment the invocation count for this index
        auto it = kDefaultActions->begin();
        std::advance(it, spec_idx);
        const std::string& spec = it->first;
        auto start = std::chrono::steady_clock::now();
        bool used_duty_cycle = false;
        {
          auto& kernels = direct_kernels::Kernels();
          auto kit = kernels.find(spec);
          if (kit != kernels.end()) {
            // Direct invocation: run until remaining slot deadline, but with
            // sub-iteration duty cycling so per-thread CPU% follows the
            // per-thread total weight (1 - no_op_ratio) instead of saturating
            // for any non-zero weight.
            auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                start.time_since_epoch()).count();
            long remaining_us = static_cast<long>(duration_us)
                              - (start_us - loop_start_time_us);
            if (remaining_us > 0) {
              double duty = std::max(0.0, std::min(1.0, 1.0 - no_op_ratio));
              direct_kernels::RunWithDutyCycle(kit->second, remaining_us, duty);
              used_duty_cycle = true;
            }
          } else {
            benchmark::RunSpecifiedBenchmarks(&display_reporter, spec);
          }
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // std::cerr << "[BM_Mimesys] Finished benchmark:" << spec << ">" << elapsed_us << std::endl;
        index_invocation_count[spec_idx] += elapsed_us;
        // Outer sleep is only needed when the kernel ran as a tight loop (i.e.
        // we did NOT do sub-iteration duty cycling above).
        long long current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
        if (!used_duty_cycle) {
          auto sleep_time = static_cast<int>(elapsed_us * no_op_ratio / (1 - no_op_ratio));
          auto slack_time = duration_us - (current_time_us - loop_start_time_us);
          sleep_time = std::min(static_cast<int>(slack_time), sleep_time);
          if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
          }
        }

        // if (PAPI_read(EventSet, values) != PAPI_OK) {
        //   std::cerr << "PAPI read counters error!" << std::endl;
        //   return;
        // }
        // num_insts = values[0];
        // prev_num_insts = num_insts;

        current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
        prev_time_diff_us = time_diff_us;
        time_diff_us = current_time_us - loop_start_time_us;

        if (time_diff_us >= duration_us) {
          // std::cerr << "[BM_Mimesys] Finished benchmark:" << spec << ">" << elapsed_us << std::endl;
          // std::cerr << "[BM_Mimesys] Sleeping for " << sleep_time << " us (no_op_ratio=" << no_op_ratio << ", spec=" << spec << ")" << std::endl;
          // std::cerr << "[BM_Mimesys] prev time diff: " << prev_time_diff_us << " us, curr time diff: " << time_diff_us << " us, " << duration_us << std::endl;
          break;
        }
      }

      // if (PAPI_read(EventSet, values) != PAPI_OK) {
      //   std::cerr << "PAPI read counters error!" << std::endl;
      //   return;
      // }
      // num_insts = values[0];
    }

    // std::cerr << "Finish Loop - time diff us: " << time_diff_us << ", duration us: " << duration_us << std::endl;

  // Stop counting instructions after the loop
  // if (PAPI_stop(EventSet, values) != PAPI_OK) {
  //     std::cerr << "PAPI stop counters error!" << std::endl;
  //     return;
  // }
}

void LoadActions(std::vector<std::string>& action_names) {
  const char* env_dir = std::getenv("ACTION_LIST_PATH");
  std::string action_list_path = env_dir ? env_dir : "mimesys_actions.txt"; // fallback if env not set

  if (!action_list_path.empty()) {
    std::ifstream infile(action_list_path);
    if (!infile) {
      std::cerr << "Failed to open action list file: " << action_list_path << std::endl;
    } else {
      std::string line;
      while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        action_names.push_back(line);
      }
    }
  }
}

void SaveProfilingResults(const std::vector<std::pair<std::string, int>>& actions) {
  const char* env_dir = std::getenv("ACTION_PROFILING_CACHE_DIR");
  std::string profiled_stats_dir = env_dir ? env_dir : "profiled_stats"; // fallback if env not set

  std::filesystem::path cache_file = std::filesystem::path(profiled_stats_dir) / "profiling_cache.txt";
  std::ofstream outfile(cache_file);
  if (!outfile) {
    std::cerr << "Failed to open profiling cache file for writing: " << cache_file << std::endl;
  } else {
    for (const auto& [action_name, elapsed_us] : actions) {
      outfile << elapsed_us << "\n";
    }
  }
}

bool LoadProfilingCache(std::vector<std::pair<std::string, int>>& actions) {
  const char* env_dir = std::getenv("ACTION_PROFILING_CACHE_DIR");
  std::string profiled_stats_dir = env_dir ? env_dir : "profiled_stats"; // fallback if env not set

  if (!profiled_stats_dir.empty()) {
    std::filesystem::path cache_file = std::filesystem::path(profiled_stats_dir) / "profiling_cache.txt";
    std::ifstream infile(cache_file);
    if (!infile) {
      std::cerr << "Failed to open profiling cache file: " << cache_file << std::endl;
      return false;
    } else {
      std::string line;
      int line_idx = 0;
      while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        int parsed_value = std::stoi(line);
        actions[line_idx].second = parsed_value;
        ++line_idx;
      }

      return true;
    }
  }
  return false;
}

void ProfileActions(const std::vector<std::string>& action_names) {
  for (const auto& action_name : action_names) {
    kDefaultActions->emplace_back(action_name, 0);
  }

  SilentReporter display_reporter;
  for (int spec_idx = 0; spec_idx < kDefaultActions->size(); ++spec_idx) {
    // Run each benchmark in the default actions list for registration.
    auto it = kDefaultActions->begin();
    std::advance(it, spec_idx);
    const std::string& spec = it->first;
    benchmark::RunSpecifiedBenchmarks(&display_reporter, spec);
  }

  if (LoadProfilingCache(*kDefaultActions)) {
    std::cerr << "Loaded profiling cache with " << kDefaultActions->size() << " actions." << std::endl;
  } else {
    // Run profiling
    for (int spec_idx = 0; spec_idx < kDefaultActions->size(); ++spec_idx) {
      // Run each benchmark in the default actions list
      // This is to ensure that the benchmarks are registered and can be run later.
      auto it = kDefaultActions->begin();
      std::advance(it, spec_idx);
      const std::string& spec = it->first;

      long long total_elapsed_us = 0;
      int num_trials = 5;
      for (int i = 0; i < num_trials; ++i) {
        auto start = std::chrono::steady_clock::now();
        benchmark::RunSpecifiedBenchmarks(&display_reporter, spec);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_elapsed_us += elapsed_us;
      }
      auto elapsed_us = total_elapsed_us / num_trials;
      (*kDefaultActions)[spec_idx].second = elapsed_us;
      std::cerr << "Profiling: " << spec_idx << ": " << elapsed_us << std::endl;
    }

    SaveProfilingResults(*kDefaultActions);
  }

  std::cerr << "kDefaultActions contents:" << std::endl;
  for (size_t i = 0; i < kDefaultActions->size(); ++i) {
    const auto& [name, value] = (*kDefaultActions)[i];
    std::cerr << "  [" << i << "] " << name << ": " << value << std::endl;
  }
}

// ── Persistent worker pool (Q2 experiment) ──────────────────────────────────
//
// Replaces the per-slot fork-and-spawn-N-threads pattern with a pool of N
// long-lived threads created once and reused across all slots / plans.
// Each worker is pinned to one CPU. Dispatch flow per slot:
//   parent: stash per-thread (execution_order, no_op_ratio, deadline) and bump
//           a sequence counter, notify_all
//   worker: wake, run RunMimesysExecutionOrder until deadline (cooperative
//           termination — no SIGKILL), signal completion
//   parent: wait until all done sequence counters catch up
//
// Tradeoffs vs fork mode:
//   - No SIGKILL timeout enforcement: relies on cooperative deadline checks
//     inside RunMimesysExecutionOrder's outer while loop. For one-hot CRC
//     plans (the Q2 target) the cooperative check is sufficient.
//   - Per-slot transitions go from ~5-10 ms (fork+spawn+join+wait) to ~10 us
//     (cv notify+wait round trip).
//   - Workers carry warm L1/TLB across slots (intentional for max CPU%).
namespace persistent_pool {

struct WorkerTask {
  bool exit = false;
  const std::vector<int>* execution_order = nullptr;
  double no_op_ratio = 0.0;
  long long deadline_us = 0;
  bool sleep_only = false;
};

class WorkerPool {
 public:
  void Init(size_t n) {
    if (workers_.size() == n) return;
    Shutdown();
    n_ = n;
    tasks_.assign(n, {});
    task_seq_.assign(n, 0);
    done_seq_.assign(n, 0);
    workers_.reserve(n);
    for (size_t j = 0; j < n; ++j) {
      workers_.emplace_back([this, j]() { WorkerLoop(j); });
    }
  }

  void DispatchSlot(const std::vector<std::vector<int>>& exec_orders,
                    const std::vector<double>& no_op_ratios,
                    long long deadline_us) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (size_t j = 0; j < n_; ++j) {
        WorkerTask t;
        t.deadline_us = deadline_us;
        t.exit = false;
        if (j < exec_orders.size() && !exec_orders[j].empty()) {
          t.execution_order = &exec_orders[j];
          t.no_op_ratio = (j < no_op_ratios.size()) ? no_op_ratios[j] : 0.0;
          t.sleep_only = false;
        } else {
          t.execution_order = nullptr;
          t.sleep_only = true;
        }
        tasks_[j] = t;
        task_seq_[j]++;
      }
    }
    cv_dispatch_.notify_all();
    std::unique_lock<std::mutex> lk(mu_);
    cv_complete_.wait(lk, [&]() {
      for (size_t j = 0; j < n_; ++j) {
        if (done_seq_[j] != task_seq_[j]) return false;
      }
      return true;
    });
  }

  void Shutdown() {
    if (workers_.empty()) return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (size_t j = 0; j < n_; ++j) {
        tasks_[j].exit = true;
        task_seq_[j]++;
      }
    }
    cv_dispatch_.notify_all();
    for (auto& t : workers_) t.join();
    workers_.clear();
    tasks_.clear();
    task_seq_.clear();
    done_seq_.clear();
    n_ = 0;
  }

  ~WorkerPool() { Shutdown(); }

 private:
  static long long NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void WorkerLoop(size_t j) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(j % std::thread::hardware_concurrency(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    // NUMA: pin allocations to the local node for this CPU. set_mempolicy is
    // per-thread on Linux; combined with the pinned affinity above, all
    // subsequent thread_local first-touches land on the local memory node.
    syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0);
    uint64_t last_seq = 0;
    while (true) {
      WorkerTask local;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_dispatch_.wait(lk, [&]() { return task_seq_[j] != last_seq; });
        last_seq = task_seq_[j];
        local = tasks_[j];
      }
      if (local.exit) {
        std::lock_guard<std::mutex> lk(mu_);
        done_seq_[j] = last_seq;
        cv_complete_.notify_one();
        return;
      }
      long long remaining = local.deadline_us - NowUs();
      if (local.sleep_only) {
        if (remaining > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(remaining));
        }
      } else if (local.execution_order && !local.execution_order->empty()
                 && remaining > 0) {
        std::unordered_map<int, int> dummy_count;
        SilentReporter dummy_reporter;
        RunMimesysExecutionOrder(
            *local.execution_order, local.no_op_ratio,
            dummy_count, dummy_reporter,
            static_cast<unsigned int>(remaining));
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        done_seq_[j] = last_seq;
      }
      cv_complete_.notify_one();
    }
  }

  size_t n_ = 0;
  std::vector<std::thread> workers_;
  std::vector<WorkerTask> tasks_;
  std::vector<uint64_t> task_seq_;
  std::vector<uint64_t> done_seq_;
  std::mutex mu_;
  std::condition_variable cv_dispatch_;
  std::condition_variable cv_complete_;
};

// Single pool shared across the lifetime of the process.
static WorkerPool g_pool;

}  // namespace persistent_pool

// Stress Emulate Benchmark
static void BM_Mimesys(benchmark::State& state) {
  std::vector<std::string> action_names;
  LoadActions(action_names);
  ProfileActions(action_names);

  // Event to count total instructions
  long long window_time_budget_us = 1'000'000; // 5 seconds in microseconds
  SilentReporter display_reporter;

  // Hash map to count how many times each index is invoked
  std::unordered_map<int, int> index_invocation_count;

  std::cerr << "Reading execution plan..." << std::endl;
  const auto &files = GetDistributionFiles("plan");
  const auto &memstrata_command_file = GetMemstrataCommandFile("memstrata");

  const char* env_dir = std::getenv("TACC_STATS_DIR");
  std::string tacc_stats_dir = env_dir ? env_dir : "/users/dhkim/HPCPerfStats/monitor/src"; // fallback if env not set

  // Start profiler in background thread
  // int background_profiler_pid = mimesys::StartProfilingBackground(1);

  for (const auto &file : files) {
    for (auto _ : state) {
      MimesysExecutionPlan plan = ReadExecutionPlanFile(file);

      if (!memstrata_command_file.empty()) {
        std::cerr << "Running memstrata command from file: " << memstrata_command_file << std::endl;
        std::filesystem::path lock_file = memstrata_command_file;
        lock_file += ".lock";
        RunMemstrataAndWait(memstrata_command_file, lock_file);
      }


      std::vector<std::vector<int>> all_num_iters;
      std::vector<std::vector<std::vector<int>>> execution_orders;
      std::vector<std::vector<double>> no_op_ratios;

      for (const auto& execution_plan_by_threads : plan.execution_plan) {
        std::cerr << "Processing execution plan for " << execution_plan_by_threads.size() << " threads." << std::endl;
        std::vector<std::vector<int>> execution_orders_by_threads;
        std::vector<double> no_op_ratios_by_threads;
        for (const auto& row : execution_plan_by_threads) {
          auto [num_iters, no_op_ratio] = GetNumBenchmarkItersFromExecutionPlan(row, window_time_budget_us);
          auto execution_order = ToExecutionOrder(num_iters);
          execution_orders_by_threads.push_back(execution_order);
          no_op_ratios_by_threads.push_back(no_op_ratio);
        }
        execution_orders.push_back(execution_orders_by_threads);
        no_op_ratios.push_back(no_op_ratios_by_threads);
        // if (execution_orders.size() > 40) {
        //   std::cerr << "Execution orders size exceeds 40. Breaking out." << std::endl;
        //   break;
        // }
      }


      auto start_time = std::chrono::steady_clock::now();

      auto start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
      auto current_time_us = start_time_us;
      // Slot duration is configurable via MIMESYS_SLOT_US (microseconds).
      // Default 2,000,000 µs (2 s/slot). Total budget scales with #slots.
      unsigned int expected_duration_us = 2000000; // default 2 s per slot
      const char* slot_env = std::getenv("MIMESYS_SLOT_US");
      if (slot_env) {
        long long parsed = std::atoll(slot_env);
        if (parsed > 0) expected_duration_us = static_cast<unsigned int>(parsed);
      }
      // expected_duration_us_total is only used for the time-diff floor
      // computation later; it tracks expected_duration_us × (size+1) per iter.
      // We initialize it for a single-iter loop and recompute below once we
      // know execution_orders.size().
      unsigned int expected_duration_us_total = expected_duration_us * 9;
      unsigned int duration_us = expected_duration_us;
      auto count = 0;


      int profiler_pid = mimesys::StartProfiling();
      if (profiler_pid < 0) {
        std::cerr << "Failed to start profiling." << std::endl;
        return;
      }

      const char* iters_env = std::getenv("MIMESYS_ITERS");
      unsigned int iteration = iters_env ? static_cast<unsigned int>(std::atoi(iters_env)) : 1;
      const char* sleep_env = std::getenv("MIMESYS_SLEEP");
      bool do_sleep = sleep_env ? (std::atoi(sleep_env) != 0) : true;

      unsigned int loop_limit = do_sleep
          ? iteration * (static_cast<unsigned int>(execution_orders.size()) + 1)
          : iteration * static_cast<unsigned int>(execution_orders.size());

      // ── Initialize persistent worker pool to max thread width seen ───────
      size_t pool_n = 0;
      for (const auto& eo : execution_orders) pool_n = std::max(pool_n, eo.size());
      if (pool_n == 0) pool_n = std::thread::hardware_concurrency();
      persistent_pool::g_pool.Init(pool_n);

      while (count < loop_limit) {
        for (size_t i = 0; i < execution_orders.size(); ++i) {
          if (duration_us > 0) {
            auto thread_start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
            // Dispatch to persistent pool; deadline is absolute time.
            long long slot_deadline_us = thread_start_time_us
                                       + static_cast<long long>(duration_us);
            persistent_pool::g_pool.DispatchSlot(
                execution_orders[i], no_op_ratios[i], slot_deadline_us);
          }

          CollectTACCStats(tacc_stats_dir);

          auto current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
          auto time_diff_us = (current_time_us - start_time_us) - expected_duration_us * (count + 1);
          count++;

          std::cerr << "Time diff us: " << time_diff_us << ", duration us: " << (current_time_us - start_time_us) << std::endl;

          // Compensate for cumulative drift but clamp to a minimum floor so
          // no slot ever gets starved to 0.  With fork/kill each slot is
          // bounded at (expected_duration_us + grace), so time_diff_us only
          // grows when CollectTACCStats / loop overhead accumulates.
          // Floor = half the expected budget (1 s for a 2 s slot).
          if (time_diff_us > 0) {
            long long adjusted = static_cast<long long>(expected_duration_us) - time_diff_us;
            // Clamp to half the expected budget so no slot is ever starved to 0.
            long long min_duration = static_cast<long long>(expected_duration_us) / 2;
            duration_us = static_cast<unsigned int>(std::max(adjusted, min_duration));
          } else {
            duration_us = expected_duration_us;
          }
        }

        if (do_sleep) {
          std::this_thread::sleep_for(std::chrono::microseconds(duration_us));
          count++;
          CollectTACCStats(tacc_stats_dir);
        }

        duration_us = expected_duration_us;

        current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      }

      // Stop the profiling process
      std::this_thread::sleep_for(std::chrono::microseconds(1000000));
      auto filename = file.filename().string();
      filename.erase(filename.find(".h5"));
      mimesys::StopProfiling(profiler_pid, filename);

      auto end_time = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
      std::cerr << "Finished execution plan: " << file << " in " << elapsed << " microseconds." << std::endl;
    }
  }

  // mimesys::StopProfilingBackground(background_profiler_pid);

  std::cerr << "Benchmark invocation counts:" << std::endl;
  for (const auto& [idx, count] : index_invocation_count) {
    auto it = kDefaultActions->begin();
    std::advance(it, idx);
    std::cerr << "  [" << idx << "] " << it->first << ": " << count << std::endl;
  }
}

void RegisterMimesysBenchmarks() {
  std::string benchmark_name = "BM_Mimesys";
  auto benchmark_fn = fleetbench::mimesys::BM_Mimesys;

  auto* benchmark = benchmark::RegisterBenchmark(benchmark_name, benchmark_fn);
  benchmark->Iterations(1);  // Set a high iteration count for stress testing.

  // Ensure PAPI is initialized before using counters
  static bool papi_initialized = false;
  if (!papi_initialized) {
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
      std::cerr << "PAPI library init error!" << std::endl;
      return;
    }
    papi_initialized = true;
  }
}

class BenchmarkRegisterer {
 public:
  BenchmarkRegisterer() {
    // Compression Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::compression::RegisterBenchmarks);
    for (const auto& [benchmark_name, benchmark_entry] : *fleetbench::compression::kDefaultBenchmarks) {
      std::string full_benchmark_name = benchmark_name;
      if (benchmark_entry.compression_level.has_value()) {
        absl::StrAppend(&full_benchmark_name, "/compression_level:",
                        benchmark_entry.compression_level.value());
      }
      if (benchmark_entry.window_log.has_value()) {
        absl::StrAppend(&full_benchmark_name,
                        "/window_log:", benchmark_entry.window_log.value());
      }
      DynamicRegistrar::Get()->AddDefaultFilter(full_benchmark_name);
    }

    // Hashing Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::hashing::RegisterBenchmarks);
    for (const auto &[benchmark_name, _] : *fleetbench::hashing::kDefaultBenchmarks) {
      DynamicRegistrar::Get()->AddDefaultFilter(benchmark_name);
    }

    // Libc Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::libc::RegisterBenchmarks);
    for (const auto &[benchmark_name, _] : *fleetbench::libc::kDefaultBenchmarks) {
      DynamicRegistrar::Get()->AddDefaultFilter(benchmark_name);
    }

    // Proto Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::proto::RegisterBenchmarks);
    DynamicRegistrar::Get()->AddDefaultFilter("BM_PROTO_Arena");

    // SIMD Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::simd::RegisterBenchmarks);
    DynamicRegistrar::Get()->AddDefaultFilter(
        ".*num_blocks:256/enable_avx512:false/flush_cache:false");

    // STL Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::stl::RegisterBenchmarks);
    // We use the fleet-wide distribution as the defaults.
    DynamicRegistrar::Get()->AddDefaultFilter("BM_CORD_Fleet");

    // Swissmap Benchmarks
    DynamicRegistrar::Get()->AddCallback(fleetbench::swissmap::RegisterColdBenchmarks);
    DynamicRegistrar::Get()->AddDefaultFilter(
        "BM_SWISSMAP_InsertHit_Cold.*::absl::flat_hash_set.*64.*set_size:64.*"
        "density:0");

    DynamicRegistrar::Get()->AddCallback(fleetbench::swissmap::RegisterHotBenchmarks);
    DynamicRegistrar::Get()->AddDefaultFilter(
        "BM_SWISSMAP_InsertHit_Hot.*::absl::flat_hash_set.*64.*set_size:64.*"
        "density:0");

    // Stress NG
    DynamicRegistrar::Get()->AddCallback(fleetbench::stress_ng::RegisterBenchmarks);
    DynamicRegistrar::Get()->AddDefaultFilter("BM_STRESS_NG_*");

    std::cerr << "Registering stress emulate benchmarks" << std::endl;
    DynamicRegistrar::Get()->AddCallback(RegisterMimesysBenchmarks);
    DynamicRegistrar::Get()->AddDefaultFilter("BM_Mimesys");
  }
};

BenchmarkRegisterer br;

}  // namespace mimesys
}  // namespace fleetbench
