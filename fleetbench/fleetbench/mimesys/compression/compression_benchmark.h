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
namespace compression {
struct DefaultBenchmarkEntry {
  std::optional<int64_t> compression_level;
  std::optional<int64_t> window_log;
  benchmark::IterationCount iteration_count;
};

// Maps the default benchmark names to their compression levels, window sizes,
// and minimum iteration counts.
inline absl::NoDestructor<absl::flat_hash_map<std::string, DefaultBenchmarkEntry>>
    kDefaultBenchmarks(
        {{"BM_COMPRESSION_Brotli_COMPRESS_Fleet",
          DefaultBenchmarkEntry{2, 18, 1}},
         {"BM_COMPRESSION_Brotli_DECOMPRESS_Fleet",
          DefaultBenchmarkEntry{2, 18, 1}},
         {"BM_COMPRESSION_Flate_COMPRESS_Fleet",
          DefaultBenchmarkEntry{6, 15, 1}},
         {"BM_COMPRESSION_Flate_DECOMPRESS_Fleet",
          DefaultBenchmarkEntry{6, 15, 1}},
         {"BM_COMPRESSION_Snappy_COMPRESS_Fleet",
          DefaultBenchmarkEntry{std::nullopt, std::nullopt, 1}},
         {"BM_COMPRESSION_Snappy_DECOMPRESS_Fleet",
          DefaultBenchmarkEntry{std::nullopt, std::nullopt, 1}},
         {"BM_COMPRESSION_ZSTD_COMPRESS_Fleet",
          DefaultBenchmarkEntry{-1, 15, 1}},
         {"BM_COMPRESSION_ZSTD_DECOMPRESS_Fleet",
          DefaultBenchmarkEntry{0, 0, 1}}});

void RegisterBenchmarks();
}  // namespace compression
}  // namespace fleetbench
