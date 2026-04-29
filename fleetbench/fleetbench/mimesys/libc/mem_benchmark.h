// Copyright 2023 The Fleetbench Authors
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

#ifndef FLEETBENCH_MIMESYS_LIBC_MEM_BENCHMARK_H_
#define FLEETBENCH_MIMESYS_LIBC_MEM_BENCHMARK_H_

#include "absl/container/flat_hash_map.h"
#include "benchmark/benchmark.h"

namespace fleetbench {
namespace libc {

// Maps the default benchmarks to their minimum iteration counts.
inline absl::NoDestructor<absl::flat_hash_map<std::string, benchmark::IterationCount>>
    kDefaultBenchmarks({{"BM_LIBC_Bcmp_Fleet_L1", 1},
                        {"BM_LIBC_Memcmp_Fleet_L1", 1},
                        {"BM_LIBC_Memcpy_Fleet_L1", 1},
                        {"BM_LIBC_Memmove_Fleet_L1", 1},
                        {"BM_LIBC_Memset_Fleet_L1", 1},
                        {"BM_LIBC_Bcmp_Fleet_Cold", 1},
                        {"BM_LIBC_Memcmp_Fleet_Cold", 1},
                        {"BM_LIBC_Memcpy_Fleet_Cold", 1},
                        {"BM_LIBC_Memmove_Fleet_Cold", 1},
                        {"BM_LIBC_Memset_Fleet_Cold", 1}});

void RegisterBenchmarks();
}  // namespace libc
}  // namespace fleetbench

#endif  // FLEETBENCH_MIMESYS_LIBC_MEM_BENCHMARK_H_
