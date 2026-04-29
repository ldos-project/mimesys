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


namespace fleetbench {
namespace hashing {
// Maps the default benchmarks to their minimum iteration counts.
// We use the fleet-wide cold distributions as the defaults.
inline absl::NoDestructor<absl::flat_hash_map<std::string, benchmark::IterationCount>>
    kDefaultBenchmarks(
        {{"BM_HASHING_Extendcrc32cinternal_Fleet_cold", 1},
         {"BM_HASHING_Computecrc32c_Fleet_cold", 1},
         {"BM_HASHING_Combine_contiguous_Fleet_cold", 1}});

void RegisterBenchmarks();
}  // namespace hashing
}  // namespace fleetbench
