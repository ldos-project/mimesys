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
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>
#include <thread>
#include <papi.h>
#include <pqos.h>
#include <immintrin.h>  // AVX intrinsics for NT-store / NT-load kernels
#include <ctime>
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
#include <iostream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fstream>
#include <glob.h>


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

// Async wrapper: kick the popen on a detached worker thread so the main slot
// loop never blocks on it. Removes the ~100 ms/slot fork+exec+sudo overhead.
// The actual `hpcperfstatsd collect` still runs in the background and writes
// to the same TACC stats log file. Per-slot ordering vs the original code is
// slightly different (the snapshot lands moments later) but per-second
// granularity in the stats log is unchanged.
void CollectTACCStatsAsync(const std::string& tacc_stats_dir) {
  std::thread([tacc_stats_dir]() {
    CollectTACCStats(tacc_stats_dir);
  }).detach();
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

// =====================================================================
// pqos CMT/MBM (LLC occupancy + memory BW + IPC) profiler — in-process libpqos.
//
// Previous implementation popen()'d the `pqos` CLI as a background 1Hz sampler.
// That had several production issues:
//   (a) sample timing was wall-clock + uncontrolled, producing 9 samples per
//       6-sec workload that didn't align with hpcperfstatsd's 6 samples;
//   (b) start-up race conditions when RMIDs were stuck from a prior session
//       (cascaded failures: every plan on c220g5-110931 wrote 162-byte log);
//   (c) resctrl filesystem auto-mount conflicting with --iface=msr;
//   (d) lock file (/run/lock/libpqos) needed manual cleanup.
//
// libpqos calls (pqos_init / pqos_mon_start / pqos_mon_poll / pqos_mon_stop)
// give us synchronous in-process control: one snapshot per call, taken at
// exactly the moment we choose (called per-slot alongside CollectTACCStatsAsync
// so the resulting series is timestamp-aligned to hpc, no warmup/teardown
// samples). pqos_mon_poll auto-tracks deltas so MBL/MBR readings are
// "cachelines since last poll".
//
// We still write the legacy pqos.log file (TIME-blocked text rows) at stop
// so the python parser (mimesys.preprocessing.pqos_parser) doesn't need any
// change. The merged-group label is "0-19" by default; PQOS_CPUS env can
// override (e.g. "0-9" / "0-19" / "0,2,4-7").
// =====================================================================
static struct pqos_mon_data g_pqos_grp;
static bool g_pqos_initialized = false;
static bool g_pqos_started = false;
static std::string g_pqos_cpu_label = "0-19";   // CORE column text for log
struct PqosSnapshot {
  double timestamp_unix;
  double llc_kb;
  double mbm_local_mb_per_s;
  double mbm_remote_mb_per_s;
  double ipc;
  uint64_t llc_misses_delta;
};
static std::vector<PqosSnapshot> g_pqos_samples;

static std::vector<unsigned> ParsePqosCpus(const std::string& s) {
  // "0-19" / "0-9,10-19" / "0,1,2"
  std::vector<unsigned> out;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t comma = s.find(',', pos);
    std::string tok = s.substr(pos,
        comma == std::string::npos ? std::string::npos : comma - pos);
    size_t dash = tok.find('-');
    if (dash == std::string::npos) {
      int v = std::atoi(tok.c_str());
      if (v >= 0) out.push_back(static_cast<unsigned>(v));
    } else {
      int lo = std::atoi(tok.substr(0, dash).c_str());
      int hi = std::atoi(tok.substr(dash + 1).c_str());
      for (int i = lo; i <= hi; ++i)
        if (i >= 0) out.push_back(static_cast<unsigned>(i));
    }
    pos = (comma == std::string::npos) ? s.size() : comma + 1;
  }
  return out;
}

int StartProfilingPqos() {
  const char* cpus_env = std::getenv("PQOS_CPUS");
  g_pqos_cpu_label = cpus_env ? cpus_env : "0-19";
  std::vector<unsigned> cores = ParsePqosCpus(g_pqos_cpu_label);
  if (cores.empty()) {
    std::cerr << "PQOS_CPUS parsed to empty CPU list" << std::endl;
    return -1;
  }

  if (!g_pqos_initialized) {
    struct pqos_config cfg = {};
    cfg.fd_log = STDOUT_FILENO;
    cfg.verbose = 0;
    int ret = pqos_init(&cfg);
    if (ret != PQOS_RETVAL_OK) {
      std::cerr << "pqos_init failed: " << ret << std::endl;
      return -1;
    }
    g_pqos_initialized = true;
  }

  // Reset all per-core RMID assignments to 0 before allocating new ones.
  // RMID state lives in CPU MSRs and survives across binary invocations, so if
  // a prior session crashed mid-monitoring (or just left RMIDs stuck), every
  // subsequent pqos_mon_start fails with status 3 (PQOS_RETVAL_RESOURCE).
  // c220g5-110931 reliably exhibits this: 100% of its chunk_3 pqos files were
  // empty across 31 rounds without this reset.
  int reset_rc = pqos_mon_reset();
  if (reset_rc != PQOS_RETVAL_OK) {
    std::cerr << "pqos_mon_reset returned " << reset_rc << " (continuing)" << std::endl;
  }

  const enum pqos_mon_event events = static_cast<enum pqos_mon_event>(
      PQOS_MON_EVENT_L3_OCCUP | PQOS_MON_EVENT_LMEM_BW |
      PQOS_MON_EVENT_RMEM_BW | PQOS_PERF_EVENT_IPC | PQOS_PERF_EVENT_LLC_MISS);

  std::memset(&g_pqos_grp, 0, sizeof(g_pqos_grp));
  int ret = pqos_mon_start(static_cast<unsigned>(cores.size()), cores.data(),
                            events, nullptr, &g_pqos_grp);
  if (ret != PQOS_RETVAL_OK) {
    std::cerr << "pqos_mon_start failed: " << ret << std::endl;
    return -1;
  }
  g_pqos_started = true;
  g_pqos_samples.clear();
  g_pqos_samples.reserve(64);
  return 0;
}

// Take one pqos snapshot at the current moment. Call alongside
// CollectTACCStatsAsync in the slot loop to get one snapshot per slot,
// perfectly aligned with hpcperfstatsd.
void CollectPqosSnapshot() {
  if (!g_pqos_started) return;
  struct pqos_mon_data* groups[1] = { &g_pqos_grp };
  int ret = pqos_mon_poll(groups, 1);
  if (ret != PQOS_RETVAL_OK) {
    std::cerr << "pqos_mon_poll failed: " << ret << std::endl;
    return;
  }
  PqosSnapshot s;
  s.timestamp_unix = std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  // values.llc is bytes; convert to KB to match pqos CLI's LLC[KB] column.
  s.llc_kb = static_cast<double>(g_pqos_grp.values.llc) / 1024.0;
  // mbm_local_delta is BYTES since last poll (libpqos already applies the
  // CPUID-discovered cacheline scale factor). Divide by 1 MB → MB per poll
  // interval ≈ MB/s when polls are ~1 sec apart.
  s.mbm_local_mb_per_s =
      static_cast<double>(g_pqos_grp.values.mbm_local_delta) /
      (1024.0 * 1024.0);
  s.mbm_remote_mb_per_s =
      static_cast<double>(g_pqos_grp.values.mbm_remote_delta) /
      (1024.0 * 1024.0);
  s.ipc = g_pqos_grp.values.ipc;
  s.llc_misses_delta = g_pqos_grp.values.llc_misses_delta;
  g_pqos_samples.push_back(s);
}

static void WritePqosLog(const std::string& path) {
  std::ofstream f(path);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << path << " for write" << std::endl;
    return;
  }
  for (const auto& s : g_pqos_samples) {
    std::time_t tt = static_cast<std::time_t>(s.timestamp_unix);
    struct tm tm_utc;
    gmtime_r(&tt, &tm_utc);
    char tsbuf[64];
    std::strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", &tm_utc);
    f << "TIME " << tsbuf << "\n";
    f << "    CORE         IPC      MISSES     LLC[KB]   MBL[MB/s]   MBR[MB/s]\n";
    // misses suffix to match CLI: "33k" / "1.2M"
    char miss[32];
    if (s.llc_misses_delta >= 1000000)
      snprintf(miss, sizeof(miss), "%.0fM",
               static_cast<double>(s.llc_misses_delta) / 1.0e6);
    else if (s.llc_misses_delta >= 1000)
      snprintf(miss, sizeof(miss), "%.0fk",
               static_cast<double>(s.llc_misses_delta) / 1.0e3);
    else
      snprintf(miss, sizeof(miss), "%llu",
               static_cast<unsigned long long>(s.llc_misses_delta));
    char row[256];
    snprintf(row, sizeof(row),
             "%9s  %10.2f  %10s  %10.1f  %10.1f  %10.1f\n",
             g_pqos_cpu_label.c_str(),
             s.ipc, miss, s.llc_kb,
             s.mbm_local_mb_per_s, s.mbm_remote_mb_per_s);
    f << row;
  }
}

void StopProfilingPqos(int /*ignored*/) {
  if (g_pqos_started) {
    pqos_mon_stop(&g_pqos_grp);
    g_pqos_started = false;
  }
  // Write the legacy CLI-compatible log so the parser doesn't change.
  const char* log_env_dir = std::getenv("TACC_STATS_LOG_DIR");
  std::string log_dir = log_env_dir ? log_env_dir : "/var/log/hpcperfstats";
  WritePqosLog(log_dir + "/pqos.log");
  // Keep g_pqos_initialized=true across plans within the same binary
  // invocation — pqos_init is idempotent for re-start, but pqos_fini fully
  // tears down. We fini only at process exit (atexit handler not needed for
  // bazel-run lifecycle: each plan is a fresh binary).
  if (g_pqos_initialized) {
    pqos_fini();
    g_pqos_initialized = false;
  }
}

// Legacy compat — StopProfiling still references this symbol but the value
// is now ignored.
static int g_pqos_profiler_pid = -1;

int StartProfiling() {
  int period = 1;
  // return StartProfilingPCM();
  // return StartProfilingTACCStats(period);
  g_pqos_profiler_pid = StartProfilingPqos();
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

  // Stop pqos and archive its log alongside the stats file as pqos-<filename>.log.
  StopProfilingPqos(g_pqos_profiler_pid);
  g_pqos_profiler_pid = -1;
  std::string pqos_src = tacc_stats_dir + "/pqos.log";
  std::filesystem::path pqos_dst =
      std::filesystem::path(target_dir) / ("pqos-" + filename + ".log");
  std::error_code ec;
  if (std::filesystem::exists(pqos_src)) {
    std::filesystem::copy_file(pqos_src, pqos_dst,
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      std::cerr << "Failed to copy pqos.log: " << ec.message() << std::endl;
    }
    std::string trunc_cmd = "sudo truncate -s 0 " + pqos_src;
    int rc = system(trunc_cmd.c_str());
    (void)rc;
  }
}

void StopProfilingBackground(int background_profiler_pid) {
  std::string cmd = "sudo kill " + std::to_string(background_profiler_pid);
  system(cmd.c_str());
}

// Maps the default benchmarks to their minimum iteration counts.
absl::NoDestructor<std::vector<std::pair<std::string, int>>> kDefaultActions(
    std::vector<std::pair<std::string, int>>{});

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

    // Sleep-matched profiling_cache (time_per_iter = each kernel's kSleepUs)
    // provides natural per-kernel residual cutoff via rounding: fixed-work
    // kernels with kSleepUs=500ms need w ≥ 0.5 to yield num_iters=1; smaller
    // weights round to 0 and are skipped without firing any usleep-blocking
    // calls. No separate min-weight threshold needed.
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
  size_t buffer_bytes = 0;
};

// Parameterized contexts so we can have multiple buffer sizes coexisting:
//   - 256 MB: original LLC-saturating version (high LLC + BW, ~25 % CPU)
//   - 16  MB: fits in LLC (25 MB), gives high CPU at low LLC occupancy
//   -  1  MB: fits in L2/L1, pure compute (highest CPU, ~zero LLC traffic)
// Each is thread_local — total memory = sum(sizes) per thread × 20 threads.
// Old default 256MB × 20 = 5 GB was a likely contributor to OS-scheduler
// starvation post-init.
static thread_local CrcCtx g_ctx_256mb;
static thread_local CrcCtx g_ctx_16mb;
static thread_local CrcCtx g_ctx_1mb;

// Cold-faithful CRC parameters (rationale).
//   - 256 MB ≫ c220g5 L3 (~25 MB): single pass evicts whole LLC at least once.
//   -  16 MB fits inside L3:        no DRAM traffic after first pass — high CPU,
//                                   ~zero LLC occupancy signal.
//   -   1 MB fits inside L2:        purest compute test, smallest memory footprint.
//   kColdStrideBytes : 64 KB per CRC call — for 256 MB version, 1000 calls
//                      span 64 MB > LLC, forcing per-call DRAM fetch.
static constexpr size_t kColdStrideBytes = 64ull * 1024;

static void InitCrcCtx(CrcCtx& ctx, size_t buffer_bytes) {
  if (ctx.init_done) return;
  ctx.buffer.assign(buffer_bytes, 'x');
  ctx.sv = absl::string_view(ctx.buffer);
  ctx.buffer_bytes = buffer_bytes;
  static const int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
  ctx.str_lengths.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    ctx.str_lengths.push_back(sizes[i % 8]);
  }
  ctx.init_done = true;
}

// Wrapper to preserve the original symbol used by the registry.
static void InitCrcCtx() { InitCrcCtx(g_ctx_256mb, 256ull * 1024 * 1024); }

static void DirectExtendCrc32c(long deadline_us) {
  InitCrcCtx();
  auto& ctx = g_ctx_256mb;
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
      start += kColdStrideBytes;  // page-stride large enough to exceed LLC
      v0 = absl::ExtendCrc32c(v0, buf);
      benchmark::DoNotOptimize(v0);
    }
  }
}

static void DirectComputeCrc32c(long deadline_us) {
  InitCrcCtx();
  auto& ctx = g_ctx_256mb;
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
      start += kColdStrideBytes;
      auto res = absl::ComputeCrc32c(buf);
      benchmark::DoNotOptimize(res);
    }
  }
}

// Smaller-buffer Crc variants — fit entirely in L3 (16 MB) or L1/L2 (1 MB).
// Per-call stride matches buffer size to avoid stride > buffer (which would
// always restart at offset 0 and degrade to one cache-line of work).
// Goal: give AL a path to the high-CPU + low-LLC corner that the 256MB
// variant never produces.
template <size_t BufferBytes>
static void DirectExtendCrc32c_Sized(long deadline_us) {
  // Pick the right thread_local context for this buffer size.
  CrcCtx& ctx = (BufferBytes == 16 * 1024 * 1024) ? g_ctx_16mb : g_ctx_1mb;
  InitCrcCtx(ctx, BufferBytes);
  constexpr size_t stride = (BufferBytes <= 1 * 1024 * 1024) ? 4 * 1024 : 32 * 1024;
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
      start += stride;
      v0 = absl::ExtendCrc32c(v0, buf);
      benchmark::DoNotOptimize(v0);
    }
  }
}

static void DirectExtendCrc32c_16MB(long d) { DirectExtendCrc32c_Sized<16ull * 1024 * 1024>(d); }
static void DirectExtendCrc32c_1MB(long d)  { DirectExtendCrc32c_Sized<1ull  * 1024 * 1024>(d); }

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

// Weight plumbed in via this thread-local; set by the dispatcher
// (RunMimesysExecutionOrder) before invoking a weight-aware kernel.
// Defaults to 1.0 so kernels that don't honor it still behave at full duty.
static thread_local double g_current_duty = 1.0;

// Weight-scaled HDD kernel: bytes_per_write = duty * MAX_SIZE.
// Sleep stays fixed, so per-thread rate ≈ (duty * MAX_SIZE) / sleep_us
// is linear in duty (weight). The dispatcher must bypass RunWithDutyCycle
// for this kernel and pass the FULL slot deadline so size scaling — not
// chunking — controls duty.
template <size_t kMaxSize, int kSleepUs, char... kTag>
static void DirectHddWeightScaledImpl(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  if (buf.empty()) {
    buf.assign(kMaxSize, 'Z');
    // Distinct per-thread-file key per template instantiation so concurrent
    // kernels don't collide on the same path.
    char tag[] = {kTag..., '\0'};
    path = PerThreadPath((std::string("hddws_") + tag).c_str());
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(512,
      static_cast<size_t>(duty * static_cast<double>(kMaxSize)));
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { usleep(kSleepUs); continue; }
    auto n = write(fd, buf.data(), target);
    benchmark::DoNotOptimize(n);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
    usleep(kSleepUs);
  }
}

// Weight-scaled READ kernel. Uses a single SHARED backing file for all
// threads (populated exactly once by the first thread to invoke this kernel
// via std::call_once). Each thread opens the file read-only and preads at
// rotating offsets; posix_fadvise(DONTNEED) evicts pages so subsequent reads
// hit the device. Shared-file means setup IO is ~256 MB once total (not
// 256 MB × 20 threads), small enough to land inside the warmup window.
template <size_t kMaxSize, int kSleepUs, char... kTag>
static void DirectHddReadWeightScaledImpl(long deadline_us) {
  static thread_local int fd = -1;
  static thread_local std::vector<char> buf;
  static thread_local size_t thread_offset = 0;
  // 256 MB backing file: big enough that the page cache can't retain it all
  // while still being achievable in a single sequential populate. The
  // populate is triggered from MAIN THREAD before the worker pool starts and
  // before the first measured slot — see PreInitOnMainThread() below — so the
  // ~2.5 s populate cost does NOT pollute steady-state IO measurements.
  static constexpr size_t kFileBytes = 256 * 1024 * 1024;
  static std::once_flag init_flag;
  static std::string shared_path;
  static int shared_keepalive_fd = -1;   // keeps the unlinked inode alive
  std::call_once(init_flag, [] {
    char tag[] = {kTag..., '\0'};
    char p[160];
    snprintf(p, sizeof(p), "/tmp/mimesys_hddread_shared_%s_%d",
             tag, static_cast<int>(getpid()));
    shared_path = p;
    int wfd = open(shared_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0) return;
    std::vector<char> initbuf(1024 * 1024, 'R');
    for (size_t i = 0; i < kFileBytes / initbuf.size(); ++i) {
      if (write(wfd, initbuf.data(), initbuf.size()) <= 0) break;
    }
    fsync(wfd);
    // Hand the fd off to a static keepalive so the inode survives the unlink
    // until the process exits (or this fd is explicitly closed).
    shared_keepalive_fd = wfd;
    unlink(shared_path.c_str());
  });
  if (fd < 0) {
    // Open the same inode via /proc/self/fd/<keepalive>; the on-disk path is
    // already unlinked but the inode is still reachable via the keepalive fd.
    char proc_path[64];
    snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", shared_keepalive_fd);
    fd = open(proc_path, O_RDONLY);
    if (fd < 0) return;
    buf.assign(kMaxSize, 0);
    // Seed per-thread starting offset so concurrent threads touch different
    // 4 KB pages and we don't hot-share an inflight read across CPUs.
    thread_offset = (static_cast<size_t>(pthread_self())
                      * (4 * 1024)) % kFileBytes;
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(512,
      static_cast<size_t>(duty * static_cast<double>(kMaxSize)));
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    if (thread_offset + target > kFileBytes) thread_offset = 0;
    posix_fadvise(fd, thread_offset, target, POSIX_FADV_DONTNEED);
    auto n = pread(fd, buf.data(), target, thread_offset);
    benchmark::DoNotOptimize(n);
    thread_offset += target;
    if (kSleepUs > 0) usleep(kSleepUs);   // 0 = no-throttle: tight read loop
  }
}

static void DirectHddRead1MB_WeightScaled50ms(long d)   { DirectHddReadWeightScaledImpl<1024 * 1024,    50000, 'r', 'm'>(d); }
static void DirectHddRead1MB_WeightScaled200ms(long d)  { DirectHddReadWeightScaledImpl<1024 * 1024,   200000, 'r', 'X'>(d); }
// Sub-10 MB/s coverage: smaller blocks + 500 ms sleep give long enough between
// reads that the kernel can actually action posix_fadvise(DONTNEED) before the
// next pread, so reads hit disk rather than being absorbed by the 187 GB page
// cache that masked the earlier 256KB_NoSleep variant.
static void DirectHddRead256KB_WeightScaled500ms(long d) { DirectHddReadWeightScaledImpl< 256 * 1024,   500000, 'r', 'Y'>(d); }
static void DirectHddRead64KB_WeightScaled500ms(long d)  { DirectHddReadWeightScaledImpl<  64 * 1024,   500000, 'r', 'Z'>(d); }
static void DirectHddRead256KB_WeightScaled50ms(long d) { DirectHddReadWeightScaledImpl< 256 * 1024,    50000, 'r', 'q'>(d); }
static void DirectHddRead1MB_NoSleep(long d)            { DirectHddReadWeightScaledImpl<1024 * 1024,        0, 'r', 'n'>(d); }
static void DirectHddRead256KB_NoSleep(long d)          { DirectHddReadWeightScaledImpl< 256 * 1024,        0, 'r', 'o'>(d); }

// Weight-scaled WRITE kernel WITHOUT fsync. Writes go to page cache (memory
// speed); kernel's writeback flushes them at burst rates beyond the device's
// sustained synchronous-write ceiling. A 1-sec window can see 300+ MB/s
// during the flush burst. Trade-off: per-iter write is fast (memory speed),
// so peak block-IO depends on the OS dirty-page flush cadence, not the
// stressor's loop rate. Periodic posix_fadvise(DONTNEED) on the prior write
// nudges the kernel to start writeback eagerly.
template <size_t kMaxSize, int kSleepUs, char... kTag>
static void DirectHddWriteNoFsyncWeightScaledImpl(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static thread_local int fd = -1;
  if (buf.empty()) {
    buf.assign(kMaxSize, 'W');
    char tag[] = {kTag..., '\0'};
    path = PerThreadPath((std::string("hddwnf_") + tag).c_str());
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    unlink(path.c_str());   // file lives until fd close
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(512,
      static_cast<size_t>(duty * static_cast<double>(kMaxSize)));
  off_t offset = 0;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    // Rewrite at offset 0 every cycle (single rolling file) so we don't
    // fragment-allocate forever; kernel writeback still sees the dirty pages.
    auto n = pwrite(fd, buf.data(), target, 0);
    benchmark::DoNotOptimize(n);
    // Hint the kernel to start writeback on this range (async, no wait).
    sync_file_range(fd, 0, target, SYNC_FILE_RANGE_WRITE);
    if (kSleepUs > 0) usleep(kSleepUs);   // 0 = no-throttle: tight write loop
  }
}

static void DirectHddWriteNF_256KB_WeightScaled50ms(long d) { DirectHddWriteNoFsyncWeightScaledImpl< 256 * 1024, 50000, 'n', 's'>(d); }
static void DirectHddWriteNF_1MB_WeightScaled50ms(long d)   { DirectHddWriteNoFsyncWeightScaledImpl<1024 * 1024, 50000, 'n', 'm'>(d); }
static void DirectHddWriteNF_1MB_WeightScaled100ms(long d)  { DirectHddWriteNoFsyncWeightScaledImpl<1024 * 1024, 100000, 'n', 'Y'>(d); }
// Sub-10 MB/s write coverage. Writes go through page cache; 500 ms between
// each per-thread write ensures kernel writeback flushes them to disk during
// the sleep, producing measurable sustained io_write at ~2-10 MB/s.
static void DirectHddWriteNF_256KB_WeightScaled500ms(long d) { DirectHddWriteNoFsyncWeightScaledImpl< 256 * 1024, 500000, 'n', 'Z'>(d); }
static void DirectHddWriteNF_64KB_WeightScaled500ms(long d)  { DirectHddWriteNoFsyncWeightScaledImpl<  64 * 1024, 500000, 'n', 'W'>(d); }
static void DirectHddWriteNF_2MB_WeightScaled50ms(long d)   { DirectHddWriteNoFsyncWeightScaledImpl<2048 * 1024, 50000, 'n', 'p'>(d); }
static void DirectHddWriteNF_4MB_WeightScaled50ms(long d)   { DirectHddWriteNoFsyncWeightScaledImpl<4096 * 1024, 50000, 'n', 'q'>(d); }
static void DirectHddWriteNF_1MB_WeightScaled25ms(long d)   { DirectHddWriteNoFsyncWeightScaledImpl<1024 * 1024, 25000, 'n', 'a'>(d); }
static void DirectHddWriteNF_1MB_WeightScaled10ms(long d)   { DirectHddWriteNoFsyncWeightScaledImpl<1024 * 1024, 10000, 'n', 'b'>(d); }
static void DirectHddWriteNF_1MB_NoSleep(long d)            { DirectHddWriteNoFsyncWeightScaledImpl<1024 * 1024,     0, 'n', 'z'>(d); }
static void DirectHddWriteNF_256KB_NoSleep(long d)          { DirectHddWriteNoFsyncWeightScaledImpl< 256 * 1024,     0, 'n', 'y'>(d); }
// BurstScaled variants: tight write loop (kSleepUs=0), but the dispatch sets
// g_current_duty to the per-stressor weight (not 1.0). Inside the kernel,
// target = duty × kMaxSize → smaller per-write size at lower weights, so the
// dirty-page creation rate scales with weight. Combined with the existing
// per_entry_sleep_us between bursts, this restores monotone dose-response for
// WRITE stressors where the writeback ceiling otherwise saturates the signal.
static void DirectHddWriteNF_1MB_BurstScaled(long d)        { DirectHddWriteNoFsyncWeightScaledImpl<1024 * 1024,     0, 'n', 'B'>(d); }
static void DirectHddWriteNF_256KB_BurstScaled(long d)      { DirectHddWriteNoFsyncWeightScaledImpl< 256 * 1024,     0, 'n', 'C'>(d); }

// HYBRID stressor: write 1 MB + sync_file_range, then a TIGHT CRC32 spin for
// kSpinUs microseconds (no usleep). Per-iter CPU = 100% (no sleep), per-iter
// IO ≈ 1 MB / (write_time + kSpinUs). At kSpinUs=30 ms: 1 MB / ~35 ms ≈ 28
// MB/s/thread. 20 threads × 28 MB/s = 560 MB/s nominal (device-cap ~290).
// CPU% stays near 100% during the CRC spin → covers the high-CPU + high-IO
// regime the throttled stressors can't reach.
template <size_t kMaxSize, int kSpinUs, char... kTag>
static void DirectHddWriteCRCSpinImpl(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::vector<char> crc_buf;
  static thread_local std::string path;
  static thread_local int fd = -1;
  static constexpr size_t kCrcBytes = 64 * 1024;
  if (buf.empty()) {
    buf.assign(kMaxSize, 'W');
    crc_buf.assign(kCrcBytes, 'C');
    char tag[] = {kTag..., '\0'};
    path = PerThreadPath((std::string("hddwcrc_") + tag).c_str());
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    unlink(path.c_str());
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(512,
      static_cast<size_t>(duty * static_cast<double>(kMaxSize)));
  auto loop_start = std::chrono::steady_clock::now();
  uint32_t crc = 0;
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    // 1 MB write to page cache + async writeback hint
    auto n = pwrite(fd, buf.data(), target, 0);
    benchmark::DoNotOptimize(n);
    sync_file_range(fd, 0, target, SYNC_FILE_RANGE_WRITE);
    // Tight CRC32 spin for kSpinUs microseconds — keeps CPU at 100%.
    auto spin_start = std::chrono::steady_clock::now();
    while (true) {
      auto spin_el = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - spin_start).count();
      if (spin_el >= kSpinUs) break;
      crc = __builtin_ia32_crc32di(crc, *reinterpret_cast<uint64_t*>(crc_buf.data() + (crc & 0xFFF8)));
      benchmark::DoNotOptimize(crc);
    }
  }
}

static void DirectHddWriteCRC_1MB_30ms(long d) { DirectHddWriteCRCSpinImpl<1024 * 1024, 30000, 'h', 'c'>(d); }
static void DirectHddWriteCRC_1MB_15ms(long d) { DirectHddWriteCRCSpinImpl<1024 * 1024, 15000, 'h', 'd'>(d); }
static void DirectHddWriteCRC_1MB_50ms(long d) { DirectHddWriteCRCSpinImpl<1024 * 1024, 50000, 'h', 'e'>(d); }

// O_DIRECT writes: bypass page cache, go straight to device. Each pwrite
// blocks until device ack — synchronous-like but no journal commit (unlike
// fsync). Per-thread throughput = device write latency * size. Many threads
// can keep multiple queued requests in flight at the device layer.
template <size_t kMaxSize, int kSleepUs, char... kTag>
static void DirectHddWriteDirectWeightScaledImpl(long deadline_us) {
  static thread_local char* buf = nullptr;
  static thread_local std::string path;
  static thread_local int fd = -1;
  static constexpr size_t kAlign = 4096;
  if (!buf) {
    char tag[] = {kTag..., '\0'};
    path = PerThreadPath((std::string("hddwd_") + tag).c_str());
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) return;
    unlink(path.c_str());
    if (posix_memalign(reinterpret_cast<void**>(&buf), kAlign, kMaxSize) != 0) {
      buf = nullptr;
      return;
    }
    memset(buf, 'D', kMaxSize);
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(kAlign,
      static_cast<size_t>(duty * static_cast<double>(kMaxSize)));
  target = (target / kAlign) * kAlign;
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    auto n = pwrite(fd, buf, target, 0);
    benchmark::DoNotOptimize(n);
    if (kSleepUs > 0) usleep(kSleepUs);
  }
}

static void DirectHddWriteD_1MB_WeightScaled50ms(long d) { DirectHddWriteDirectWeightScaledImpl<1024 * 1024, 50000, 'd', 'm'>(d); }
static void DirectHddWriteD_1MB_WeightScaled25ms(long d) { DirectHddWriteDirectWeightScaledImpl<1024 * 1024, 25000, 'd', 'a'>(d); }
static void DirectHddWriteD_1MB_WeightScaled0ms(long d)  { DirectHddWriteDirectWeightScaledImpl<1024 * 1024,     0, 'd', '0'>(d); }

// MEMSET-STREAM kernel: repeatedly memset a per-thread buffer too large to fit
// in L3 (32 MB vs ~14 MB LLC). Each memset evicts L3 and writes back to DRAM,
// generating measurable memory_bandwidth_write traffic. Round-0 v2 coverage
// showed only SwissMap_Insert produces >5% bw_write; this kernel fills that
// gap without depending on the allocator. Per-thread buffer = no cross-thread
// cache contention. 20 threads × 32 MB = 640 MB total memory footprint.
template <size_t kBufSize, int kSleepUs, char... kTag>
static void DirectMemsetStreamImpl(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local int tick = 0;
  if (buf.empty()) buf.assign(kBufSize, 0);
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(64,
      static_cast<size_t>(duty * static_cast<double>(kBufSize)));
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    std::memset(buf.data(), static_cast<int>(tick++ & 0xFF), target);
    benchmark::DoNotOptimize(buf.data());
    if (kSleepUs > 0) usleep(kSleepUs);
  }
}

static void DirectMemset_32MB_NoSleep(long d)          { DirectMemsetStreamImpl<32 * 1024 * 1024,     0, 'm', 'z'>(d); }
static void DirectMemset_32MB_WeightScaled50ms(long d) { DirectMemsetStreamImpl<32 * 1024 * 1024, 50000, 'm', 'a'>(d); }

// STREAM-SIMD kernel: tight FMA-style loop over a per-thread aligned double
// buffer. Compiler with -O3 -mavx512f vectorizes the loop; each iteration
// loads 8 doubles, performs y = a*x + b (FMA), stores back. This achieves
// HIGH IPC (≈2-3, the only stressor in our panel doing this) WHILE driving
// real memory bandwidth (buffer >> L3 → DRAM-bound). Coverage gap: the
// previous panel had no workload in the (high IPC × high mem BW) quadrant
// because vectorized + memory-bound is rare.
//   32 MB buffer (4M doubles) >> 14 MB LLC → DRAM streaming
//    8 MB buffer (1M doubles) ≈ LLC-resident → mid mem BW, high L3 traffic
template <size_t kBufBytes, int kSleepUs, char... kTag>
static void DirectStreamSIMDImpl(long deadline_us) {
  static thread_local std::vector<double> buf;
  static constexpr size_t kN = kBufBytes / sizeof(double);
  if (buf.empty()) {
    buf.assign(kN, 1.0);
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(8, static_cast<size_t>(duty * static_cast<double>(kN)));
  // Round target down to multiple of 8 so vectorized inner loop is full-width.
  target = (target / 8) * 8;
  double a = 1.000001;  // not exactly 1 so the compiler can't optimize away
  double b = 0.0000001;
  auto loop_start = std::chrono::steady_clock::now();
  double* __restrict__ p = buf.data();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    // y = a*x + b — vectorized by the compiler to vfmadd231pd over zmm regs.
    for (size_t i = 0; i < target; ++i) {
      p[i] = a * p[i] + b;
    }
    benchmark::DoNotOptimize(p);
    if (kSleepUs > 0) usleep(kSleepUs);
  }
}

static void DirectStreamSIMD_32MB_NoSleep(long d) { DirectStreamSIMDImpl<32 * 1024 * 1024, 0, 's', 'z'>(d); }
static void DirectStreamSIMD_8MB_NoSleep(long d)  { DirectStreamSIMDImpl< 8 * 1024 * 1024, 0, 's', 'a'>(d); }
static void DirectStreamSIMD_4MB_NoSleep(long d)  { DirectStreamSIMDImpl< 4 * 1024 * 1024, 0, 's', '4'>(d); }

// ──────────────────────────────────────────────────────────────────────────
// MEMCPY-STREAM kernel: per-thread src + dst buffers. memcpy generates BOTH
// reads (src) and writes (dst), giving symmetric mid-BW when buffer ≥ LLC.
// Fills the empty (BW_rd in [1,7] AND BW_wr in [1,7]) corner identified
// in the v2 pool's coverage analysis.
template <size_t kBufBytes, int kSleepUs, char... kTag>
static void DirectMemcpyStreamImpl(long deadline_us) {
  static thread_local std::vector<char> src;
  static thread_local std::vector<char> dst;
  if (src.empty()) { src.assign(kBufBytes, 'A'); dst.assign(kBufBytes, 0); }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(64,
      static_cast<size_t>(duty * static_cast<double>(kBufBytes)));
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    std::memcpy(dst.data(), src.data(), target);
    benchmark::DoNotOptimize(dst.data());
    if (kSleepUs > 0) usleep(kSleepUs);
  }
}

static void DirectMemcpy_8MB_NoSleep(long d)  { DirectMemcpyStreamImpl< 8 * 1024 * 1024, 0, 'p', 'a'>(d); }
static void DirectMemcpy_32MB_NoSleep(long d) { DirectMemcpyStreamImpl<32 * 1024 * 1024, 0, 'p', 'z'>(d); }

// ──────────────────────────────────────────────────────────────────────────
// NT-STORE kernel: write-only DRAM traffic via _mm256_stream_si256. NT stores
// bypass cache, so the read side is essentially zero — fills the (BW_wr in
// [1,7] AND BW_rd < 1) corner. K=1 yields ~3 GB/s write, K=4 ~12 GB/s, scales
// linearly until memory controller saturation.
template <size_t kBufBytes, int kSleepUs, char... kTag>
static void DirectNTStoreImpl(long deadline_us) {
  static thread_local char* buf = nullptr;
  if (!buf) {
    if (posix_memalign(reinterpret_cast<void**>(&buf), 32, kBufBytes) != 0) return;
    std::memset(buf, 0, kBufBytes);
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(32,
      static_cast<size_t>(duty * static_cast<double>(kBufBytes)));
  target = (target / 32) * 32;
  const __m256i val = _mm256_set1_epi64x(static_cast<long long>(0xCAFEBABEull));
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    __m256i* p = reinterpret_cast<__m256i*>(buf);
    const size_t n_blocks = target / 32;
    for (size_t i = 0; i < n_blocks; ++i) {
      _mm256_stream_si256(p + i, val);
    }
    _mm_sfence();
    benchmark::DoNotOptimize(buf);
    if (kSleepUs > 0) usleep(kSleepUs);
  }
}

static void DirectNTStore_8MB_NoSleep(long d)  { DirectNTStoreImpl< 8 * 1024 * 1024, 0, 'n', 'a'>(d); }
static void DirectNTStore_32MB_NoSleep(long d) { DirectNTStoreImpl<32 * 1024 * 1024, 0, 'n', 'z'>(d); }

// ──────────────────────────────────────────────────────────────────────────
// SCAN kernel: read-only DRAM via _mm256_stream_load_si256 (NT load) over a
// per-thread buffer. Reads are streaming (cache-bypassing on supported
// platforms), so writes ≈ 0 and reads scale with K_thread. Pairs with NT-store
// to give us independent rd-only and wr-only axes.
template <size_t kBufBytes, int kSleepUs, char... kTag>
static void DirectScanImpl(long deadline_us) {
  static thread_local char* buf = nullptr;
  if (!buf) {
    if (posix_memalign(reinterpret_cast<void**>(&buf), 32, kBufBytes) != 0) return;
    std::memset(buf, 'S', kBufBytes);
  }
  double duty = std::max(0.0, std::min(1.0, g_current_duty));
  if (duty < 0.005) {
    std::this_thread::sleep_for(std::chrono::microseconds(deadline_us));
    return;
  }
  size_t target = std::max<size_t>(32,
      static_cast<size_t>(duty * static_cast<double>(kBufBytes)));
  target = (target / 32) * 32;
  const size_t n_blocks = target / 32;
  __m256i acc = _mm256_setzero_si256();
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    const __m256i* p = reinterpret_cast<const __m256i*>(buf);
    for (size_t i = 0; i < n_blocks; ++i) {
      __m256i v = _mm256_stream_load_si256(p + i);
      acc = _mm256_xor_si256(acc, v);
    }
    benchmark::DoNotOptimize(acc);
    if (kSleepUs > 0) usleep(kSleepUs);
  }
}

static void DirectScan_8MB_NoSleep(long d)  { DirectScanImpl< 8 * 1024 * 1024, 0, 'r', 'a'>(d); }
static void DirectScan_32MB_NoSleep(long d) { DirectScanImpl<32 * 1024 * 1024, 0, 'r', 'z'>(d); }

// Four candidates spanning the IO bandwidth space (max size × sleep_us):
//   4KB   / 10 ms : LIGHT       — peak ~  8 MB/s @ 20 threads
//  16KB   /  5 ms : LIGHT-MID   — peak ~ 50 MB/s
// 256KB   / 50 ms : MID         — peak ~ 75 MB/s
//   1MB   / 50 ms : HEAVY       — peak ~ 85 MB/s (device-capped)
static void DirectHdd4KB_WeightScaled10ms(long d)   { DirectHddWeightScaledImpl<     4 * 1024,  10000, '4', 'k'>(d); }
static void DirectHdd16KB_WeightScaled5ms(long d)   { DirectHddWeightScaledImpl<    16 * 1024,   5000, '1', '6'>(d); }
static void DirectHdd256KB_WeightScaled50ms(long d) { DirectHddWeightScaledImpl<   256 * 1024,  50000, '2', '5'>(d); }
static void DirectHdd1MB_WeightScaled50ms(long d)   { DirectHddWeightScaledImpl<1024 * 1024,   50000, '1', 'm'>(d); }

// ~16 MB/s per thread saturated: 1 MB write + 50 ms sleep + fsync.
// Per-iter time ≈ 10 ms write/fsync + 50 ms sleep = ~60 ms.
// Goal: linear-in-weight IO with NO 1-MB minimum spike at low weights.
static void DirectHdd1MB_Throttled50ms(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static constexpr size_t kSize = 1024 * 1024;
  static constexpr int kSleepUs = 50000;
  if (buf.empty()) {
    buf.assign(kSize, 'Z');
    path = PerThreadPath("hdd1m_t50");
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { usleep(kSleepUs); continue; }
    auto n = write(fd, buf.data(), buf.size());
    benchmark::DoNotOptimize(n);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
    usleep(kSleepUs);
  }
}

// ── Low-IO throttled kernels (fill the [1000, 10000) sectors/s gap) ─────────
//
// Stable, near-linear-in-weight IO bandwidth — each iteration is a fixed-size
// write followed by a fixed sleep, so IO rate per thread is roughly
//   bytes/iter / (write_latency + sleep_us).
// Aggregated across N threads, total IO scales linearly with N.

// ~3.2 MB/s per thread: 16 KB write + 5 ms sleep + fsync.
// Sectors/s per thread ≈ 6.4k → in the missing [1000, 10000) bucket.
static void DirectHdd16KB_Throttled5ms(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static constexpr size_t kSize = 16 * 1024;
  static constexpr int kSleepUs = 5000;
  if (buf.empty()) {
    buf.assign(kSize, 'Z');
    path = PerThreadPath("hdd16k_t5");
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { usleep(kSleepUs); continue; }
    auto n = write(fd, buf.data(), buf.size());
    benchmark::DoNotOptimize(n);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
    usleep(kSleepUs);
  }
}

// ~400 KB/s per thread: 4 KB write + 10 ms sleep + fsync.
// Sectors/s per thread ≈ 800 → fills the [100, 1000) bucket.
static void DirectHdd4KB_Throttled10ms(long deadline_us) {
  static thread_local std::vector<char> buf;
  static thread_local std::string path;
  static constexpr size_t kSize = 4 * 1024;
  static constexpr int kSleepUs = 10000;
  if (buf.empty()) {
    buf.assign(kSize, 'Z');
    path = PerThreadPath("hdd4k_t10");
  }
  auto loop_start = std::chrono::steady_clock::now();
  while (true) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - loop_start).count();
    if (elapsed >= deadline_us) break;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { usleep(kSleepUs); continue; }
    auto n = write(fd, buf.data(), buf.size());
    benchmark::DoNotOptimize(n);
    fsync(fd);
    close(fd);
    unlink(path.c_str());
    usleep(kSleepUs);
  }
}


using KernelFn = std::function<void(long)>;
static const std::unordered_map<std::string, KernelFn>& Kernels() {
  static const std::unordered_map<std::string, KernelFn> m = {
    // BM_HASHING_*_cold direct kernels — see InitCrcCtx() for the cold-faithful
    // rewrite (256 MB buffer + 64 KB stride). Framework fallback is not viable
    // here because GetNumBenchmarkItersFromExecutionPlan returns num_iters=0
    // (kDefaultActions->second is left at 0 by ProfileActions), so the
    // framework path runs zero iterations.
    {"BM_HASHING_Extendcrc32cinternal_Fleet_cold", DirectExtendCrc32c},
    {"BM_HASHING_Computecrc32c_Fleet_cold",        DirectComputeCrc32c},
    // High-CPU / low-LLC variants: 16 MB fits in L3, 1 MB in L2.
    // Same hashing op as the 256 MB version, no DRAM traffic after warmup.
    {"BM_HASHING_Extendcrc32c_Fleet_L3_16MB",      DirectExtendCrc32c_16MB},
    {"BM_HASHING_Extendcrc32c_Fleet_L2_1MB",       DirectExtendCrc32c_1MB},
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
    {"BM_STRESS_NG_Readahead",                 DirectReadahead},
    {"BM_STRESS_NG_Fallocate_4MB",             DirectFallocate4MB},
    {"BM_STRESS_NG_Fallocate_256KB",           DirectFallocate256KB},
    {"BM_STRESS_NG_Hdd_1MB",                   DirectHdd1MB},
    {"BM_STRESS_NG_Hdd_1MB_Throttled_50ms",    DirectHdd1MB_Throttled50ms},
    {"BM_STRESS_NG_Hdd_4KB_WeightScaled_10ms",   DirectHdd4KB_WeightScaled10ms},
    {"BM_STRESS_NG_Hdd_16KB_WeightScaled_5ms",   DirectHdd16KB_WeightScaled5ms},
    {"BM_STRESS_NG_Hdd_256KB_WeightScaled_50ms", DirectHdd256KB_WeightScaled50ms},
    {"BM_STRESS_NG_Hdd_1MB_WeightScaled_50ms",   DirectHdd1MB_WeightScaled50ms},
    {"BM_STRESS_NG_HddRead_1MB_WeightScaled_50ms",   DirectHddRead1MB_WeightScaled50ms},
    {"BM_STRESS_NG_HddRead_1MB_WeightScaled_200ms",  DirectHddRead1MB_WeightScaled200ms},
    {"BM_STRESS_NG_HddRead_256KB_WeightScaled_500ms", DirectHddRead256KB_WeightScaled500ms},
    {"BM_STRESS_NG_HddRead_64KB_WeightScaled_500ms",  DirectHddRead64KB_WeightScaled500ms},
    {"BM_STRESS_NG_HddRead_256KB_WeightScaled_50ms", DirectHddRead256KB_WeightScaled50ms},
    {"BM_STRESS_NG_HddRead_1MB_NoSleep",             DirectHddRead1MB_NoSleep},
    {"BM_STRESS_NG_HddRead_256KB_NoSleep",           DirectHddRead256KB_NoSleep},
    {"BM_STRESS_NG_HddWriteNF_256KB_NoSleep",        DirectHddWriteNF_256KB_NoSleep},
    {"BM_STRESS_NG_HddWriteNF_256KB_WeightScaled_50ms", DirectHddWriteNF_256KB_WeightScaled50ms},
    {"BM_STRESS_NG_HddWriteNF_1MB_WeightScaled_50ms", DirectHddWriteNF_1MB_WeightScaled50ms},
    {"BM_STRESS_NG_HddWriteNF_1MB_WeightScaled_100ms", DirectHddWriteNF_1MB_WeightScaled100ms},
    {"BM_STRESS_NG_HddWriteNF_256KB_WeightScaled_500ms", DirectHddWriteNF_256KB_WeightScaled500ms},
    {"BM_STRESS_NG_HddWriteNF_64KB_WeightScaled_500ms",  DirectHddWriteNF_64KB_WeightScaled500ms},
    {"BM_STRESS_NG_HddWriteNF_2MB_WeightScaled_50ms", DirectHddWriteNF_2MB_WeightScaled50ms},
    {"BM_STRESS_NG_HddWriteNF_4MB_WeightScaled_50ms", DirectHddWriteNF_4MB_WeightScaled50ms},
    {"BM_STRESS_NG_HddWriteNF_1MB_WeightScaled_25ms", DirectHddWriteNF_1MB_WeightScaled25ms},
    {"BM_STRESS_NG_HddWriteNF_1MB_WeightScaled_10ms", DirectHddWriteNF_1MB_WeightScaled10ms},
    {"BM_STRESS_NG_HddWriteNF_1MB_NoSleep",           DirectHddWriteNF_1MB_NoSleep},
    {"BM_STRESS_NG_HddWriteNF_1MB_BurstScaled",       DirectHddWriteNF_1MB_BurstScaled},
    {"BM_STRESS_NG_HddWriteNF_256KB_BurstScaled",     DirectHddWriteNF_256KB_BurstScaled},
    {"BM_STRESS_NG_HddWriteCRC_1MB_30ms",             DirectHddWriteCRC_1MB_30ms},
    {"BM_STRESS_NG_HddWriteCRC_1MB_15ms",             DirectHddWriteCRC_1MB_15ms},
    {"BM_STRESS_NG_HddWriteCRC_1MB_50ms",             DirectHddWriteCRC_1MB_50ms},
    {"BM_STRESS_NG_HddWriteD_1MB_WeightScaled_50ms",  DirectHddWriteD_1MB_WeightScaled50ms},
    {"BM_STRESS_NG_HddWriteD_1MB_WeightScaled_25ms",  DirectHddWriteD_1MB_WeightScaled25ms},
    {"BM_STRESS_NG_HddWriteD_1MB_WeightScaled_0ms",   DirectHddWriteD_1MB_WeightScaled0ms},
    {"BM_DIRECTMEMSET_32MB_NoSleep",                 DirectMemset_32MB_NoSleep},
    {"BM_DIRECTSTREAMSIMD_32MB_NoSleep",             DirectStreamSIMD_32MB_NoSleep},
    {"BM_DIRECTSTREAMSIMD_8MB_NoSleep",              DirectStreamSIMD_8MB_NoSleep},
    {"BM_DIRECTSTREAMSIMD_4MB_NoSleep",              DirectStreamSIMD_4MB_NoSleep},
    // New mid-BW kernels for the 1-7 GB/s coverage hole.
    {"BM_DIRECTMEMCPY_8MB_NoSleep",                  DirectMemcpy_8MB_NoSleep},
    {"BM_DIRECTMEMCPY_32MB_NoSleep",                 DirectMemcpy_32MB_NoSleep},
    {"BM_DIRECTNTSTORE_8MB_NoSleep",                 DirectNTStore_8MB_NoSleep},
    {"BM_DIRECTNTSTORE_32MB_NoSleep",                DirectNTStore_32MB_NoSleep},
    {"BM_DIRECTSCAN_8MB_NoSleep",                    DirectScan_8MB_NoSleep},
    {"BM_DIRECTSCAN_32MB_NoSleep",                   DirectScan_32MB_NoSleep},
    {"BM_DIRECTMEMSET_32MB_WeightScaled_50ms",       DirectMemset_32MB_WeightScaled50ms},
    {"BM_STRESS_NG_Hdd_64KB",                  DirectHdd64KB},
    {"BM_STRESS_NG_Hdd_16KB_Throttled_5ms",    DirectHdd16KB_Throttled5ms},
    {"BM_STRESS_NG_Hdd_4KB_Throttled_10ms",    DirectHdd4KB_Throttled10ms},
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

    // ── Fair per-entry time slicing ────────────────────────────────────────
    // Each entry in execution_order gets a proportional slice of slot time
    // instead of consuming the full remaining deadline (the prior behavior
    // let the FIRST entry eat the whole slot). The slice is sized so that
    // ONE pass through execution_order fills the slot exactly:
    //   per_entry_active_us = duration_us * duty / n_entries
    //   per_entry_sleep_us  = duration_us * (1 - duty) / n_entries
    // This makes mixed-stressor actions actually run all the stressors with
    // shares matching their weights (which num_iters now encodes via the
    // ratio*100 scaling for direct kernels).
    auto time_diff_us = 0LL;
    auto prev_time_diff_us = 0LL;
    {
      const size_t n_entries = execution_order.size();
      if (n_entries == 0) return;
      const double duty = std::max(0.0, std::min(1.0, 1.0 - no_op_ratio));
      const long per_entry_active_us = static_cast<long>(std::round(
          static_cast<double>(duration_us) * duty / static_cast<double>(n_entries)));
      const long per_entry_sleep_us = static_cast<long>(std::round(
          static_cast<double>(duration_us) * (1.0 - duty) / static_cast<double>(n_entries)));

      // Per-stressor weight = count(spec_idx in execution_order) / 100. Used by
      // *_BurstScaled kernels which scale target = weight × kMaxSize so that
      // dirty-page / read-throughput rate scales with weight (option 2 — fixes
      // the write-saturation problem where NoSleep variants peg at the
      // writeback ceiling regardless of weight).
      std::unordered_map<int, double> per_spec_weight;
      for (int idx : execution_order) per_spec_weight[idx] += 1.0;
      for (auto& kv : per_spec_weight) kv.second /= 100.0;

      for (int spec_idx : execution_order) {
        auto now = std::chrono::steady_clock::now();
        long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - loop_start_time).count();
        long remaining = static_cast<long>(duration_us) - static_cast<long>(elapsed);
        if (remaining <= 0) break;

        auto it = kDefaultActions->begin();
        std::advance(it, spec_idx);
        const std::string& spec = it->first;
        auto& kernels = direct_kernels::Kernels();
        auto kit = kernels.find(spec);
        long this_active = std::min<long>(per_entry_active_us, remaining);

        auto kstart = std::chrono::steady_clock::now();
        if (kit != kernels.end()) {
          if (spec.find("BurstScaled") != std::string::npos) {
            // Per-stressor weight controls request size (target = w × kMaxSize)
            // so dirty-page / read rate scales with weight even with no in-kernel
            // sleep. The dispatch-level per_entry_sleep_us still throttles bursts.
            auto wit = per_spec_weight.find(spec_idx);
            direct_kernels::g_current_duty = (wit != per_spec_weight.end())
                ? wit->second : 1.0;
            kit->second(this_active);
          } else if (spec.find("WeightScaled") != std::string::npos) {
            // We already gave the kernel its exact per-entry time share.
            // Set g_current_duty = 1.0 so the kernel uses its FULL configured
            // write size during that share.
            direct_kernels::g_current_duty = 1.0;
            kit->second(this_active);
          } else {
            // Direct kernel: run for the per-entry share. Bypass the inner
            // RunWithDutyCycle since we've already computed the exact active
            // time; idle time is folded into the per_entry_sleep_us below.
            kit->second(this_active);
          }
        } else {
          benchmark::RunSpecifiedBenchmarks(&display_reporter, spec);
        }
        long long kelapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - kstart).count();
        index_invocation_count[spec_idx] += kelapsed;

        // Distribute idle time evenly between entries so the duty cycle
        // pattern stays smooth (no end-of-slot burst sleep).
        long long now_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - loop_start_time).count();
        long remaining_after_busy = static_cast<long>(duration_us) - static_cast<long>(now_elapsed);
        long this_sleep = std::min<long>(per_entry_sleep_us, remaining_after_busy);
        if (this_sleep > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(this_sleep));
        }
      }
      return;   // exit the function (the legacy while loop below is bypassed)
    }
    // ── Legacy path (unreachable; kept temporarily as a guard) ─────────────
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
            auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                start.time_since_epoch()).count();
            long remaining_us = static_cast<long>(duration_us)
                              - (start_us - loop_start_time_us);
            if (remaining_us > 0) {
              double duty = std::max(0.0, std::min(1.0, 1.0 - no_op_ratio));
              if (spec.find("WeightScaled") != std::string::npos) {
                direct_kernels::g_current_duty = duty;
                kit->second(remaining_us);
              } else {
                direct_kernels::RunWithDutyCycle(kit->second, remaining_us, duty);
              }
              used_duty_cycle = true;
            }
          } else {
            benchmark::RunSpecifiedBenchmarks(&display_reporter, spec);
          }
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        index_invocation_count[spec_idx] += elapsed_us;
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

      // For direct kernels (the common case in our setup), skip framework
      // profiling — RunSpecifiedBenchmarks measures the slow framework path
      // (e.g., ~4 s for cold CRC), which produces a num_iters value out of
      // proportion with the per-iter time of OTHER direct kernels. The result
      // is that mixed-stressor actions get a distorted weight ratio in
      // execution_order. Assign a uniform short time_per_iter so num_iters
      // (computed in GetNumBenchmarkItersFromExecutionPlan as round(target_time
      // / time_per_iter)) scales cleanly with the requested weight.
      auto& dk = direct_kernels::Kernels();
      if (dk.count(spec)) {
        it->second = 5000;   // 5 ms — close to the slowest direct kernel's
                              // single-iter time (HddWrite_1MB ≈ 5 ms), so a
                              // per-entry budget of 5 ms cleanly fits ONE iter
                              // of any direct kernel.
        continue;
      }

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
  size_t num_workers() const { return n_; }
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
      long long _wake_us = NowUs();
      long long remaining = local.deadline_us - _wake_us;
      long long _wake_latency = local.deadline_us - static_cast<long long>(local.deadline_us); // 0 baseline
      // Worker reads deadline; how much time elapsed from main setting deadline (deadline_us - duration_us) to now (_wake_us)?
      // We can't know duration without it being stored, so just report remaining.
      long long _kernel_start = _wake_us;
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
      long long _kernel_end = NowUs();
      if (j == 0 && !local.sleep_only) {
        std::cerr << "WORKER0 wake_us=" << _wake_us
                  << " deadline=" << local.deadline_us
                  << " remaining_given=" << remaining << "us"
                  << " kernel_ran=" << (_kernel_end - _kernel_start) << "us"
                  << " late_vs_deadline=" << (_kernel_end - local.deadline_us) << "us"
                  << std::endl;
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

      // Sync moved to right before StartProfiling (post-warmup). See below.

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


      // Profiling start moved to AFTER warmup (below) so sample 0 lands at
      // the workload start (first plan step), not 5s earlier during warmup.
      int profiler_pid = -1;

      const char* iters_env = std::getenv("MIMESYS_ITERS");
      unsigned int iteration = iters_env ? static_cast<unsigned int>(std::atoi(iters_env)) : 1;
      const char* sleep_env = std::getenv("MIMESYS_SLEEP");
      bool do_sleep = sleep_env ? (std::atoi(sleep_env) != 0) : true;

      unsigned int loop_limit = do_sleep
          ? iteration * (static_cast<unsigned int>(execution_orders.size()) + 1)
          : iteration * static_cast<unsigned int>(execution_orders.size());

      // ── Main-thread pre-init: trigger lazy setup (file populate, big buf
      //    alloc) for IO-resident direct kernels BEFORE the worker pool
      //    spins up. Running it from the main thread sequentializes the
      //    expensive populates (e.g. 256 MB shared-file write for the
      //    HddRead kernels) into a single block before any timed work
      //    begins; we then trigger a TACC stats sample at warmup-end so the
      //    first measured delta starts AFTER the populate is done.
      //    We invoke each registered IO kernel with duty=0 and a 1 ms
      //    deadline so it just runs its call_once init and returns.
      {
        double saved = direct_kernels::g_current_duty;
        direct_kernels::g_current_duty = 0.0;
        for (const auto& kv : direct_kernels::Kernels()) {
          const std::string& name = kv.first;
          if (name.find("HddRead") == std::string::npos &&
              name.find("HddWriteNF") == std::string::npos &&
              name.find("WeightScaled") == std::string::npos) continue;
          kv.second(1000);  // 1 ms; triggers std::call_once if present
        }
        direct_kernels::g_current_duty = saved;
      }

      // ── Initialize persistent worker pool to max thread width seen ───────
      size_t pool_n = 0;
      for (const auto& eo : execution_orders) pool_n = std::max(pool_n, eo.size());
      if (pool_n == 0) pool_n = std::thread::hardware_concurrency();
      persistent_pool::g_pool.Init(pool_n);

      // ── Pin main thread away from worker CPUs (last logical core).
      //    Without this, main thread + worker 0 contend for CPU 0 and the
      //    measured per-core utilization caps around 89% even at long slots.
      {
        cpu_set_t main_set;
        CPU_ZERO(&main_set);
        unsigned int main_cpu = std::thread::hardware_concurrency() - 1;
        CPU_SET(main_cpu, &main_set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_set);
      }

      // ── Kernel pre-init pass: trigger each IO-resident direct kernel's
      //    first-call setup (file allocation, shared-file populate, big buf
      //    alloc) BEFORE the timed steady-state loop. Without this, the first
      //    slot that hits a previously-unused weight-scaled IO kernel pays a
      //    ~hundred-MB write tax for setup that pollutes the IO measurement.
      //    We dispatch one slot per kernel-name with all 20 worker positions
      //    at duty=1.0 and a 20 ms deadline — each kernel returns quickly
      //    once its lazy-init is done; threads that don't need any setup
      //    just spin briefly.
      {
        // Save current duty so we can restore (we set 1.0 here)
        double saved = direct_kernels::g_current_duty;
        direct_kernels::g_current_duty = 1.0;
        auto& kernels = direct_kernels::Kernels();
        for (int spec_idx = 0; spec_idx < static_cast<int>(kDefaultActions->size()); ++spec_idx) {
          auto it = kDefaultActions->begin();
          std::advance(it, spec_idx);
          const std::string& spec = it->first;
          // Pre-init only the kernels that allocate disk-backed state.
          if (spec.find("HddRead") == std::string::npos &&
              spec.find("HddWriteNF") == std::string::npos &&
              spec.find("WeightScaled") == std::string::npos) continue;
          if (kernels.find(spec) == kernels.end()) continue;
          // Per-worker execution_order = [spec_idx]; no-op ratio 0.
          const size_t nw = persistent_pool::g_pool.num_workers();
          std::vector<std::vector<int>> init_orders(nw, std::vector<int>{spec_idx});
          std::vector<double> init_noop(nw, 0.0);
          long long init_deadline = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count() + 50000;  // 50 ms
          persistent_pool::g_pool.DispatchSlot(init_orders, init_noop, init_deadline);
        }
        direct_kernels::g_current_duty = saved;
      }

      // ── Warmup dispatch: triggers worker-pool first-dispatch cv chain,
      //    forces each worker's CPU pinning + set_mempolicy syscalls to run,
      //    and warms per-worker thread_local kernel state by running the
      //    first window's execution plan.
      //    HPCPerfStats sampler captures its first %begin row ~1.4s after
      //    binary launch (one full sample interval). Holding the warmup for
      //    at least that long ensures the first REAL slot (i=0) begins right
      //    after that first sample boundary, so /proc/stat deltas measure
      //    busy time exclusively from the steady-state slot loop. Result:
      //    no startup-window artifact dragging the first per-core %CPU below
      //    the steady-state 99.8%.
      if (!execution_orders.empty()) {
        // Extended 5 s warmup gives a wide buffer for the main-thread populate
        // tail and writeback flushes to settle before the timed loop.
        // (Earlier we also called CollectTACCStatsAsync at warmup-end to
        // create a clean baseline paragraph, but that shifted the trace by
        // one delta and broke the dataloader's `_prepend_zero_metric_slot`
        // cycle-2 alignment — curr/sleep got swapped. Reverted.)
        long long warmup_deadline = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() + 5000000;  // 5s
        persistent_pool::g_pool.DispatchSlot(
            execution_orders[0], no_op_ratios[0], warmup_deadline);
      }

      // (sync-fix) Barrier with the target side, AFTER our calibration + worker
      // pool init + warmup but BEFORE StartProfiling.  Writes "1" to <file>.lock
      // ("synth post-warmup, ready") and polls for "2" ("target ready, go").
      // No-op when memstrata_commands/ has no file.
      if (!memstrata_command_file.empty()) {
        std::cerr << "[sync] post-warmup barrier: " << memstrata_command_file << std::endl;
        std::filesystem::path lock_file = memstrata_command_file;
        lock_file += ".lock";
        RunMemstrataAndWait(memstrata_command_file, lock_file);
      }

      // Gate the binary's own hpcperfstatsd usage behind MIMESYS_INTERNAL_PROFILING.
      // When this var is unset or "0", we skip StartProfiling / CollectTACCStatsAsync
      // / StopProfiling so they don't fight with mimebench's host-side samplers over
      // /var/log/hpcperfstats/current (which they were doing — see the conflict
      // documented in the run-time docs).  Default off; old behavior is opt-in via
      // MIMESYS_INTERNAL_PROFILING=1.
      const char* prof_env = std::getenv("MIMESYS_INTERNAL_PROFILING");
      bool do_internal_profiling = prof_env && std::atoi(prof_env) != 0;
      if (do_internal_profiling) {
        profiler_pid = mimesys::StartProfiling();
        if (profiler_pid < 0) {
          std::cerr << "Failed to start profiling." << std::endl;
          return;
        }
      } else {
        std::cerr << "[mimesys] internal profiling disabled "
                  << "(set MIMESYS_INTERNAL_PROFILING=1 to re-enable)." << std::endl;
      }

      // Reset slot-loop clock AFTER warmup so the drift-compensation below
      // doesn't see "+10s behind" and clamp the next ~20 slots to half-duration.
      start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();

      while (count < loop_limit) {
        for (size_t i = 0; i < execution_orders.size(); ++i) {
          auto _t_pre_dispatch = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
          if (duration_us > 0) {
            auto thread_start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
            // Dispatch to persistent pool; deadline is absolute time.
            long long slot_deadline_us = thread_start_time_us
                                       + static_cast<long long>(duration_us);
            persistent_pool::g_pool.DispatchSlot(
                execution_orders[i], no_op_ratios[i], slot_deadline_us);
          }
          auto _t_post_dispatch = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();

          // Gated on MIMESYS_INTERNAL_PROFILING.  See block above.
          // hpc + pqos snapshots taken at the SAME moment so the two series
          // align 1:1 (no timestamp games downstream).
          if (do_internal_profiling) {
            CollectTACCStatsAsync(tacc_stats_dir);
            mimesys::CollectPqosSnapshot();
          }
          auto _t_post_collect = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
          std::cerr << "SLOT_PROF i=" << i
                    << " dispatch=" << (_t_post_dispatch - _t_pre_dispatch) << "us"
                    << " collect=" << (_t_post_collect - _t_post_dispatch) << "us"
                    << " duration_us=" << duration_us
                    << " over=" << (long long)(_t_post_dispatch - _t_pre_dispatch) - (long long)duration_us << "us"
                    << std::endl;

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
          if (do_internal_profiling) {
            CollectTACCStatsAsync(tacc_stats_dir);
            mimesys::CollectPqosSnapshot();
          }
        }

        duration_us = expected_duration_us;

        current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      }

      // Stop the profiling process — gated to match the gate above.
      if (do_internal_profiling) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000000));
        auto filename = file.filename().string();
        filename.erase(filename.find(".h5"));
        mimesys::StopProfiling(profiler_pid, filename);
      }

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

    // SIMD Benchmarks (disabled: simd_library is clang-only and not needed for
    // the fleetbench mimesys action list — skip to avoid an undefined-symbol
    // link error when building with gcc).
    // DynamicRegistrar::Get()->AddCallback(fleetbench::simd::RegisterBenchmarks);
    // DynamicRegistrar::Get()->AddDefaultFilter(
    //     ".*num_blocks:256/enable_avx512:false/flush_cache:false");

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

// ── /tmp temp-file leak prevention ─────────────────────────────────────────
// Direct IO kernels create /tmp/mimesys_*_<id> backing files. They unlink the
// path immediately after open so the inode is freed on fd close — but if the
// process is SIGKILL'd between open() and unlink(), the file is orphaned.
// Repeated kills accumulate ~256 MB per leak and fill the worker disk
// (observed: 6/8 workers hit 100 % full after a debugging session).
//
// Two-layer defense:
//   (1) Startup sweep: at static init, remove any /tmp/mimesys_*_<pid>
//       file whose owning <pid> is no longer alive.
//   (2) Exit handlers (atexit + SIGTERM/SIGINT): unlink anything still
//       matching /tmp/mimesys_*_<my_pid> on the way out.
namespace {

static void UnlinkMyTempFiles() {
  glob_t g;
  // Patterns covering all temp file conventions in this binary:
  //   /tmp/mimesys_hddread_shared_<tag>_<pid>
  //   /tmp/mimesys_direct_<tag>_<tid>       (per-thread, no pid)
  //   /tmp/mimesys_hddread_*_<tid>          (per-thread, no pid)
  // Match by glob and unlink everything we can — file is per-process, but
  // pthread_self() is reused only inside this proc, so cleaning unconditionally
  // at our exit is safe (we own all the open fds).
  const char* patterns[] = {
    "/tmp/mimesys_*",
  };
  for (const char* pat : patterns) {
    if (glob(pat, 0, NULL, &g) == 0) {
      for (size_t i = 0; i < g.gl_pathc; ++i) {
        unlink(g.gl_pathv[i]);
      }
      globfree(&g);
    }
  }
}

static void StartupOrphanCleanup() {
  int my_pid = getpid();
  glob_t g;
  if (glob("/tmp/mimesys_hddread_shared_*", 0, NULL, &g) != 0) return;
  int unlinked = 0;
  for (size_t i = 0; i < g.gl_pathc; ++i) {
    const char* path = g.gl_pathv[i];
    const char* uscore = strrchr(path, '_');
    if (!uscore) continue;
    int owner_pid = atoi(uscore + 1);
    if (owner_pid <= 0 || owner_pid == my_pid) continue;
    // kill(pid, 0) returns 0 if alive, -1/ESRCH if dead.
    if (kill(owner_pid, 0) != 0 && errno == ESRCH) {
      if (unlink(path) == 0) unlinked++;
    }
  }
  globfree(&g);
  if (unlinked > 0) {
    std::cerr << "[cleanup] removed " << unlinked
              << " orphaned /tmp/mimesys_hddread_shared_* files" << std::endl;
  }
}

struct TempFileCleaner {
  TempFileCleaner() {
    StartupOrphanCleanup();
    std::atexit(UnlinkMyTempFiles);
    // No SIGTERM/SIGINT handlers — they were interfering with Google
    // Benchmark's internal signal management and breaking multi-threaded
    // slot dispatch. SIGKILL leaks are caught by the startup sweep above
    // on the next run.
  }
};
static TempFileCleaner _temp_file_cleaner_;

}  // anonymous namespace

}  // namespace mimesys
}  // namespace fleetbench
