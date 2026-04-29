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
#include <unistd.h>
#include <sys/ioctl.h>
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
        benchmark::RunSpecifiedBenchmarks(&display_reporter, spec);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // std::cerr << "[BM_Mimesys] Finished benchmark:" << spec << ">" << elapsed_us << std::endl;
        index_invocation_count[spec_idx] += elapsed_us;
        auto sleep_time = static_cast<int>(elapsed_us * no_op_ratio / (1 - no_op_ratio));
        auto current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
        auto slack_time = duration_us - (current_time_us - loop_start_time_us);
        sleep_time = std::min(static_cast<int>(slack_time), sleep_time);
        if (sleep_time > 0) {
          // Sleep to simulate the no-op ratio
          // std::cerr << "[BM_Mimesys] Sleeping for " << sleep_time << " us (no_op_ratio=" << no_op_ratio << ", spec=" << spec << ")" << std::endl;
          std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
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
      unsigned int expected_duration_us_total = 18000000; // 18 seconds
      unsigned int expected_duration_us = 2000000; // 2 seconds per slot
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
      while (count < loop_limit) {
        for (size_t i = 0; i < execution_orders.size(); ++i) {
          if (duration_us > 0) {
            auto thread_start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();

            // Fork a child process to run this slot so the parent can
            // SIGKILL it if it overruns the time budget.
            pid_t child_pid = fork();
            if (child_pid == 0) {
              // ── Child: spawn worker threads and run the slot ────────────
              // Set NUMA memory policy to local-node allocation once for the
              // whole child process; inherited by all threads below.
              syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0);

              std::vector<std::thread> threads;
              for (size_t j = 0; j < execution_orders[i].size(); ++j) {
                threads.emplace_back([&, i, j, thread_start_time_us]() {
                  cpu_set_t cpuset;
                  CPU_ZERO(&cpuset);
                  CPU_SET(j % std::thread::hardware_concurrency(), &cpuset);
                  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                  std::unordered_map<int, int> child_index_invocation_count;
                  SilentReporter child_reporter;

                  auto thread_current_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                  auto thread_start_time_diff_us = thread_current_time_us - thread_start_time_us;

                  if (!execution_orders[i][j].empty()) {
                    RunMimesysExecutionOrder(
                      execution_orders[i][j], no_op_ratios[i][j],
                      child_index_invocation_count, child_reporter,
                      duration_us - thread_start_time_diff_us
                    );
                  } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(
                      duration_us - thread_start_time_diff_us));
                  }
                });
              }
              for (auto& t : threads) t.join();
              _exit(0);

            } else if (child_pid > 0) {
              // ── Parent: poll until child exits or deadline passes ───────
              // Allow a 100 ms grace period before issuing SIGKILL so that
              // slots that finish just barely over budget still exit cleanly.
              static constexpr long long kGracePeriodUs = 100000LL;
              long long deadline_us = thread_start_time_us
                                      + static_cast<long long>(duration_us)
                                      + kGracePeriodUs;
              while (true) {
                int status;
                pid_t r = waitpid(child_pid, &status, WNOHANG);
                if (r > 0) break;   // child finished naturally
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now().time_since_epoch()).count();
                if (now_us >= deadline_us) {
                  kill(child_pid, SIGKILL);
                  waitpid(child_pid, nullptr, 0);
                  std::cerr << "[BM_Mimesys] Force-killed slot " << i
                            << " — overran budget by "
                            << (now_us - thread_start_time_us
                                - static_cast<long long>(duration_us))
                            << " us" << std::endl;
                  break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
              }

            } else {
              std::cerr << "[BM_Mimesys] fork() failed for slot " << i
                        << ": " << std::strerror(errno) << std::endl;
            }
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
