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

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "fleetbench/common/common.h"
#include "fleetbench/dynamic_registrar.h"
#include "fleetbench/mimesys/swissmap/swissmap_benchmark.h"

#include "absl/base/attributes.h"
#include "absl/random/random.h"

// All benchmarks in this file are for cold lookups.
namespace fleetbench {

namespace swissmap {

using ::benchmark::DoNotOptimize;

// Helper function used to implement two similar benchmarks that the given input
// key is NOT present in the set.
template <template <class...> class SetT, size_t kValueSizeT, bool kLookup>
static void FindMiss_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the colder the benchmark and the longer it takes
  // to run.
  static constexpr size_t kMinTotalBytes = 256 << 10;

  auto& sc = SetsCache<Set>::GetInstance();
  std::vector<Set>& cached_sets =
      sc.GetGeneratedSets(state.range(0), kMinTotalBytes / kValueSizeT,
                          static_cast<Density>(state.range(1)));
  std::vector<uint32_t>& keys = sc.GetNonExistingKeys(cached_sets);

  // If kLookup is false, we need to create a copy of the cached sets vector
  // because the benchmark loop modifies it.
  std::conditional_t<kLookup, std::vector<Set>&, std::vector<Set>> sets =
      cached_sets;

  int warmup = 5;
  while (true) {
    for (uint32_t key : keys) {
      if (--warmup < 0 && !state.KeepRunningBatch(sets.size())) return;
      for (Set& set : sets) {
        DoNotOptimize(set);
        DoNotOptimize(key);
        if (kLookup) {
          auto res = set.find(key);
          DoNotOptimize(res);
        } else {
          auto res = set.insert(key);
          DoNotOptimize(res);
        }
      }
      if (!kLookup) {
        // PauseTiming()/ResumeTiming() are relatively expensive, but it is OK
        // to use them here because `sets` is large, and thus the cost of the
        // two functions is small compared to the cost of the loop above.
        if (warmup < 0) state.PauseTiming();
        for (Set& set : sets) {
          set.erase(key);
        }
        if (warmup < 0) state.ResumeTiming();
      }
    }
  }
}

// Measures the time it takes to `find` an existent element.
//
// assert(set.find(key) == set.end());
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_FindMiss_Cold(benchmark::State& state) {
  return FindMiss_Cold<SetT, kValueSizeT, /*kLookup=*/true>(state);
}

// Measures the time it takes to `insert` an existent element.
//
//  assert(set.insert(key).second);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertMiss_Cold(benchmark::State& state) {
  return FindMiss_Cold<SetT, kValueSizeT, /*kLookup=*/false>(state);
}

// Helper function used to implement two similar benchmarks defined below that
// the given input key is present in the set.
template <template <class...> class SetT, size_t kValueSizeT, class Lookup>
void LookupHit_Cold(benchmark::State& state, Lookup lookup) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the colder the benchmark and the longer it takes to
  // run.
  static constexpr size_t kMinTotalBytes = 256 << 10;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& sets = sc.GetGeneratedSets(state.range(0), kMinTotalBytes / kValueSizeT,
                                   static_cast<Density>(state.range(1)));
  auto& keys = sc.GetTransposedRandomizedKeys(sets);
  auto& n_sets_of_size = sc.GetNumSetsOfSize(sets);

  int64_t warmup = 5;
  while (true) {
    for (size_t i = 0; i != GetLargestSetSize(sets); ++i) {
      if ((warmup-- <= 0) && !state.KeepRunningBatch(n_sets_of_size[i + 1]))
        return;
      for (size_t j = 0; j < n_sets_of_size[i + 1]; ++j) {
        lookup(&sets[j], keys[i * sets.size() + j]);
      }
    }
  }
}

// Measures the time it takes to `find` an existent element.
//
//   asssert(set.find(key) != set.end());
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_FindHit_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;
  return LookupHit_Cold<SetT, kValueSizeT>(state, [](Set* set, uint32_t key) {
    DoNotOptimize(set);
    DoNotOptimize(key);
    auto res = set->find(key);
    DoNotOptimize(res);
  });
}

// Measures the time it takes to `insert` an existent element.
//
//   assert(!set.insert(key).second);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertHit_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;
  return LookupHit_Cold<SetT, kValueSizeT>(state, [](Set* set, uint32_t key) {
    DoNotOptimize(set);
    DoNotOptimize(key);
    auto res = set->insert(key);
    DoNotOptimize(res);
  });
}

// Measures the time it takes to iterate over a set and read its every element.
// The reported time is per element. In other words, the pseudo code below
// counts as `set.size()` iterations.
//
//   for (const auto& elem : set) {
//     Read(elem);
//   }
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_Iterate_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the colder the benchmark and the longer it takes
  // to run.
  static constexpr size_t kMinTotalBytes = 1 << 20;
  static constexpr size_t kStride = 16;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& sets = sc.GetGeneratedSets(state.range(0), kMinTotalBytes / kValueSizeT,
                                   static_cast<Density>(state.range(1)));

  // `sets.back()` has the minimum size.
  const size_t num_strides = sets.back().size() / kStride;
  assert(num_strides > 0);

  size_t total_num_sets = sets.size();
  std::vector<std::vector<typename Set::const_iterator>> set_iterators;
  set_iterators.reserve(total_num_sets);

  alignas(Value<kValueSizeT>) char data[kValueSizeT];

  while (true) {
    set_iterators.clear();
    // This construction ensures cold setup for each of the generated sets.
    for (const Set& set : sets) {
      auto it = set.begin();
      std::vector<typename Set::const_iterator> iters;
      iters.reserve(num_strides);
      for (size_t i = 0; i != num_strides; ++i) {
        iters.push_back(it);
        std::advance(it, kStride);
      }
      set_iterators.push_back(std::move(iters));
    }
    for (size_t i = 0; i != kStride; ++i) {
      for (size_t j = 0; j != num_strides; ++j) {
        if (!state.KeepRunningBatch(total_num_sets)) return;
        // Iterate over sets in the inner loop to reduce caching and ensure cold
        // environment.
        for (size_t k = 0; k != total_num_sets; ++k) {
          std::vector<typename Set::const_iterator>& curr_set_iterators =
              set_iterators[k];
          auto& iter = curr_set_iterators[j];
          memcpy(data, &*iter, kValueSizeT);
          DoNotOptimize(data);
          ++iter;
        }
      }
    }
  }
}

// Microbenchmarks below exercise behavior in pathological conditions.

// Measures the time it takes to `erase` an existent element and then `insert`
// a new element. The element is re-inserted immediately after erase to
// prevent sets from shrinking. Newly inserted element is different from
// existent, to ensure tombstones are created by implementations that use
// them.
//
// Depending on the set implementation, erased elements may create tombstones
// which affect performance on insertion and frequency of rehashing, which is
// what this microbenchmark is capturing.
//
// Only operates on low density hashset. It may in the future be extended to
// also run for high density.
//
//   assert(set.erase(key1));
//   assert(set.insert(key2).second);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_EraseInsert_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the colder the benchmark and the longer it takes
  // to run.
  static constexpr size_t kMinTotalBytes = 1 << 20;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& sets = sc.GetGeneratedSets(state.range(0), kMinTotalBytes / kValueSizeT,
                                   Density::kMin);
  auto& keys = sc.GetExtendedKeys(sets);
  const size_t largest_set_size = GetLargestSetSize(sets);

  while (true) {
    // We create a copy of 'sets' so that the condition that existent elements
    // are erased holds in every iteration of the outer loop.
    std::vector<Set> sets_copy = sets;
    for (size_t i = 0; i != largest_set_size; ++i) {
      // Iterate over sets in the inner loop to reduce caching and ensure cold
      // environment.
      if (!state.KeepRunningBatch(sets_copy.size())) return;
      for (size_t j = 0; j != sets_copy.size(); ++j) {
        Set& curr_set = sets_copy[j];
        // skip over out-of-bounds access for smaller sets
        if (i >= curr_set.size()) continue;
        auto erase_result = curr_set.erase(keys[j][i]);
        DoNotOptimize(erase_result);
        auto insert_result = curr_set.insert(keys[j][i + curr_set.size()]);
        DoNotOptimize(insert_result);
      }
    }
  }
}

// Measures the time it takes to `clear` a set and then `insert` the same
// elements in the order they were in the set. The reported time is per
// element. In other words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.insert(key1);
//   ...
//   set.insert(keyN);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertManyOrdered_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the colder the benchmark and the longer it takes
  // to run.
  static constexpr size_t kMinTotalBytes = 1 << 20;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& cached_sets =
      sc.GetGeneratedSets(state.range(0), kMinTotalBytes / kValueSizeT,
                          static_cast<Density>(state.range(1)));
  auto& keys = sc.GetTransposedKeys(cached_sets);
  auto& n_sets_of_size = sc.GetNumSetsOfSize(cached_sets);

  // create a copy because 'sets' is modified in the loop below
  auto sets = cached_sets;
  size_t largest_set_size = GetLargestSetSize(sets);

  while (true) {
    for (Set& set : sets) {
      set.erase(set.begin(), set.end());
    }
    for (size_t i = 0; i != largest_set_size; ++i) {
      if (!state.KeepRunningBatch(n_sets_of_size[i + 1])) return;
      for (size_t j = 0; j < n_sets_of_size[i + 1]; ++j) {
        auto insert_result = sets[j].insert(keys[i * sets.size() + j]);
        DoNotOptimize(insert_result);
      }
    }
  }
}

// Measures the time it takes to `clear` a set and then `insert` the same
// elements back in random order. The reported time is per element. In other
// words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.insert(key1);
//   ...
//   set.insert(keyN);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertManyUnordered_Cold(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the colder the benchmark and the longer it takes
  // to run.
  static constexpr size_t kMinTotalBytes = 1 << 20;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& cached_sets =
      sc.GetGeneratedSets(state.range(0), kMinTotalBytes / kValueSizeT,
                          static_cast<Density>(state.range(1)));
  auto& keys = sc.GetTransposedRandomizedKeys(cached_sets);
  auto& n_sets_of_size = sc.GetNumSetsOfSize(cached_sets);

  // create a copy because 'sets' is modified in the loop below
  auto sets = cached_sets;
  size_t largest_set_size = GetLargestSetSize(sets);

  while (true) {
    for (Set& set : sets) {
      set.erase(set.begin(), set.end());
    }
    for (size_t i = 0; i != largest_set_size; ++i) {
      if (!state.KeepRunningBatch(n_sets_of_size[i + 1])) return;
      for (size_t j = 0; j < n_sets_of_size[i + 1]; ++j) {
        auto insert_result = sets[j].insert(keys[i * sets.size() + j]);
        DoNotOptimize(insert_result);
      }
    }
  }
}

void RegisterColdBenchmarks() {
  std::vector<benchmark::internal::Benchmark*> benchmarks;
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindMiss_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindMiss_Cold, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertMiss_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertMiss_Cold, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindHit_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindHit_Cold, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertHit_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertHit_Cold, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_Iterate_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_Iterate_Cold, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks,
                                  BM_SWISSMAP_InsertManyOrdered_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks,
                                  BM_SWISSMAP_InsertManyOrdered_Cold, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks,
                                  BM_SWISSMAP_InsertManyUnordered_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks,
                                  BM_SWISSMAP_InsertManyUnordered_Cold, 64);
  for (auto* benchmark : benchmarks) {
    benchmark->ArgNames({"set_size", "density"});
    // If UseExplicitIterationCounts() is false, then the code below is
    // equivalent to:
    //     benchmark->Ranges({
    //         {1 << 4, 1 << 20},
    //         {static_cast<int64_t>(Density::kMin),
    //          static_cast<int64_t>(Density::kMax)},
    //     });
    // We cannot use the same approach if UseExplicitIterationCounts() is true
    // because it is not possible to set different iteration counts for
    // benchmarks that are part of the same family. We therefore manually do
    // what `Ranges()` does, and register a separate benchmark for the special
    // case for which we want to set an explicit iteration count.
    for (int64_t set_size : {16, 64, 512, 4096, 32768, 262144, 1048576}) {
      for (int64_t density = static_cast<int64_t>(Density::kMin);
           density <= static_cast<int64_t>(Density::kMax); density++) {
        if (UseExplicitIterationCounts() &&
            absl::string_view(benchmark->GetName()) ==
                "BM_SWISSMAP_InsertHit_Cold<::absl::flat_hash_set, 64>" &&
            set_size == 64 && density == static_cast<int64_t>(Density::kMin)) {
          REGISTER_BENCHMARK_TEMPLATE(BM_SWISSMAP_InsertHit_Cold,
                                      ::absl::flat_hash_set, 64)
              ->ArgNames({"set_size", "density"})
              ->Args({set_size, density})
              ->Iterations(1);
        } else {
          benchmark->Args({set_size, density});
          benchmark->Iterations(1);
        }
      }
    }
  }

  std::vector<benchmark::internal::Benchmark*> erase_insert_benchmarks;
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(erase_insert_benchmarks,
                                  BM_SWISSMAP_EraseInsert_Cold, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(erase_insert_benchmarks,
                                  BM_SWISSMAP_EraseInsert_Cold, 64);
  for (auto* benchmark : erase_insert_benchmarks) {
    benchmark->ArgNames({"set_size"})
        ->Ranges({
            {1 << 4, 1 << 20},
        });
    benchmark->Iterations(1);
  }
}

// Measures the time it takes to `find` a non-existent element.
//
// assert(set.find(key) == set.end());
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_FindMiss_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the less the results will depend on randomness and
  // the longer the benchmark will run.
  static constexpr size_t kMinTotalKeyCount = 64 << 10;
  // The larger this value, the hotter the benchmark and the longer it will
  // run.
  static constexpr size_t kOpsPerKey = 512;

  auto& sc = SetsCache<Set>::GetInstance();
  std::vector<Set>& sets = sc.GetGeneratedSets(
      state.range(0), kMinTotalKeyCount, static_cast<Density>(state.range(1)));
  const size_t keys_per_set = kMinTotalKeyCount / sets.size();

  while (state.KeepRunningBatch(sets.size() * keys_per_set * kOpsPerKey)) {
    for (auto& set : sets) {
      for (size_t i = 0; i != keys_per_set; ++i) {
        uint32_t key = RandomNonexistent();
        for (size_t j = 0; j != kOpsPerKey; ++j) {
          DoNotOptimize(set);
          DoNotOptimize(key);
          auto res = set.find(key);
          DoNotOptimize(res);
        }
      }
    }
  }
}

// Measures the time it takes to `insert` a non-existent element.
//
//  assert(set.insert(key).second);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertMiss_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the less the results will depend on randomness and
  // the longer the benchmark will run.
  static constexpr size_t kMinTotalKeyCount = 64 << 10;

  auto& sc = SetsCache<Set>::GetInstance();
  // We need to create a copy of the cached sets vector
  // because the benchmark loop modifies it.
  std::vector<Set> sets = sc.GetGeneratedSets(
      state.range(0), kMinTotalKeyCount, static_cast<Density>(state.range(1)));
  const size_t keys_per_set = kMinTotalKeyCount / sets.size();

  std::vector<uint32_t> keys;
  keys.resize(keys_per_set);
  for (uint32_t& key : keys) key = RandomNonexistent();

  while (state.KeepRunningBatch(sets.size() * keys_per_set)) {
    for (auto& set : sets) {
      for (uint32_t key : keys) {
        DoNotOptimize(set);
        DoNotOptimize(key);
        auto res = set.insert(key);
        DoNotOptimize(res);
      }
    }

    // Since we are using the same set for all iterations, we need to reset it
    // to avoid `InsertHit` behavior.
    state.PauseTiming();
    for (auto& set : sets) {
      for (uint32_t& key : keys) {
        set.erase(key);
      }
    }
    state.ResumeTiming();
  }
}

// Helper function used to implement two similar benchmarks defined below that
// the given input key is present in the set.
template <template <class...> class SetT, size_t kValueSizeT, class Lookup>
void LookupHit_Hot(benchmark::State& state, Lookup lookup) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  static constexpr size_t kMinTotalKeyCount = 64 << 10;
  static constexpr size_t kOpsPerKey = 16;

  auto& sc = SetsCache<Set>::GetInstance();
  // Create a copy because 'sets' may be modified by the code below.
  std::vector<Set> sets = sc.GetGeneratedSets(
      state.range(0), kMinTotalKeyCount, static_cast<Density>(state.range(1)));

  if (sets.size() == 1) {
    // Make sure this executes for long enough by adding additional keys and
    // randomize to make it more robust.
    std::vector<uint32_t> keys = ToVector(sets.front());
    std::shuffle(keys.begin(), keys.end(), GetRNG());
    keys.resize(kMinTotalKeyCount);
    Set key_set(keys.begin(), keys.end());
    sets[0] = key_set;
  }

  while (
      state.KeepRunningBatch(sets.size() * sets.front().size() * kOpsPerKey)) {
    for (auto& set : sets) {
      for (uint32_t key : set) {
        for (size_t i = 0; i != kOpsPerKey; ++i) {
          lookup(&set, key);
        }
      }
    }
  }
}

// Measures the time it takes to `find` an existent element.
//
//   asssert(set.find(key) != set.end());
// template <class Set>
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_FindHit_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;
  return LookupHit_Hot<SetT, kValueSizeT>(state, [](Set* set, uint32_t key) {
    DoNotOptimize(set);
    DoNotOptimize(key);
    auto res = set->find(key);
    DoNotOptimize(res);
  });
}

// Measures the time it takes to `insert` an existent element.
//
//   assert(!set.insert(key).second);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertHit_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;
  return LookupHit_Hot<SetT, kValueSizeT>(state, [](Set* set, uint32_t key) {
    DoNotOptimize(set);
    DoNotOptimize(key);
    auto res = set->insert(key);
    DoNotOptimize(res);
  });
}

// Measures the time it takes to iterate over a set and read its every element.
// The reported time is per element. In other words, the pseudo code below
// counts as `set.size()` iterations.
//
//   for (const auto& elem : set) {
//     Read(elem);
//   }
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_Iterate_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the hotter the benchmark and the longer it will
  // run.
  static constexpr size_t kRepetitions = 1;
  // The larger this value, the less the results will depend on randomness and
  // the longer the benchmark will run.
  static constexpr size_t kMinTotalKeyCount = 256 << 20;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& sets = sc.GetGeneratedSets(state.range(0), kMinTotalKeyCount,
                                   static_cast<Density>(state.range(1)));
  alignas(Value<kValueSizeT>) char data[kValueSizeT];
  while (state.KeepRunningBatch(sets.size() * sets.front().size() *
                                kRepetitions)) {
    for (const auto& set : sets) {
      for (size_t i = 0; i != kRepetitions; ++i) {
        for (const auto& elem : set) {
          memcpy(data, &elem, kValueSizeT);
          DoNotOptimize(data);
        }
      }
    }
  }
}

// Microbenchmarks below exercise behavior in pathological conditions.

// Measures the time it takes to `erase` an existent element and then `insert`
// a new element. The element is re-inserted immediately after erase to prevent
// sets from shrinking. Newly inserted element is different from existent, to
// ensure tombstones are created by implementations that use them.
//
// Depending on the set implementation, erased elements may create tombstones
// which affect performance on insertion and frequency of rehashing, which is
// what this microbenchmark is capturing.
//
// Only operates on low density hashset. It may in the future be extended to
// also run for high density.
//
//   CHECK(set.erase(key1));
//   CHECK(set.insert(key2).second);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_EraseInsert_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The larger this value, the less the results will depend on randomness and
  // the longer the benchmark will run.
  static constexpr size_t kMinKeyCount = 1 << 20;

  Set s = GenerateSet<Set>(state.range(0), Density::kMin);
  const size_t set_size = s.size();
  std::vector<uint32_t> keys = ToVector(s);
  std::shuffle(keys.begin(), keys.end(), GetRNG());

  // Generate unique keys that haven't been inserted into original set before.
  absl::flat_hash_set<uint32_t> extra_keys;
  while (keys.size() < kMinKeyCount || keys.size() < 3 * s.size()) {
    uint32_t key = RandomExistent();
    // Generate a unique key that hasn't been inserted before.
    if (!s.count(key) && extra_keys.insert(key).second) keys.push_back(key);
  }

  const size_t keys_size_effective = keys.size();

  for (size_t i = 0; i != set_size; ++i) {
    // We create some overlap (i.e., keys[i] == keys[keys_size_effective+i]) for
    // 0 <= i < set_size), so that the logic in the main loop below becomes
    // simpler and does not require potentially expensive modulo operations.
    keys.push_back(keys[i]);
  }

  while (state.KeepRunningBatch(keys_size_effective)) {
    for (size_t i = 0; i != keys_size_effective; ++i) {
      DoNotOptimize(s);
      auto erase_result = s.erase(keys[i]);
      DoNotOptimize(erase_result);
      auto insert_result = s.insert(keys[i + set_size]);
      DoNotOptimize(insert_result);
    }
  }
}

// Measures the time it takes to `clear` a set and then `insert` the same
// elements in the order they were in the set. The reported time is per element.
// In other words, the pseudo code below counts as N iterations.
//
//   set.clear();
//   set.insert(key1);
//   ...
//   set.insert(keyN);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertManyOrdered_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  // The higher the value, the less contribution std::shuffle makes. The price
  // is longer benchmarking time. With 64 std::shuffle adds around 0.3 ns to
  // the benchmark results.
  static constexpr size_t kRepetitions = 1;
  // The larger this value, the less the results will depend on randomness and
  // the longer the benchmark will run.
  static constexpr size_t kMinTotalKeyCount = 256 << 10;

  auto& sc = SetsCache<Set>::GetInstance();
  auto& cached_sets = sc.GetGeneratedSets(state.range(0), kMinTotalKeyCount,
                                          static_cast<Density>(state.range(1)));
  auto& keys = sc.GetKeys(cached_sets);

  // create a copy because 'sets' is modified in the loop below
  auto sets = cached_sets;

  while (state.KeepRunningBatch(sets.size() * sets.front().size() *
                                kRepetitions)) {
    for (size_t i = 0; i != sets.size(); ++i) {
      for (size_t j = 0; j != kRepetitions; ++j) {
        sets[i].erase(sets[i].begin(), sets[i].end());
        for (uint32_t key : keys[i]) {
          auto res = sets[i].insert(key);
          DoNotOptimize(res);
        }
      }
    }
  }
}

template <class KeysGenerator, class EmptySetGetter>
static void RunInsertManyUnordered_Hot(benchmark::State& state,
                                       KeysGenerator keys_gen,
                                       EmptySetGetter empty_set_getter) {
  // The higher the value, the less contribution
  // state.{PauseTiming/ResumeTiming} makes.
  static constexpr size_t kRepetitions = 1;
  // The larger this value, the less the results will depend on randomness.
  static constexpr size_t kMinInsertions = 256 << 10;

  std::vector<uint32_t> keys = keys_gen();
  const size_t n = std::max(size_t{1}, kMinInsertions / keys.size());
  while (state.KeepRunningBatch(keys.size() * n * kRepetitions)) {
    for (size_t i = 0; i != n; ++i) {
      state.PauseTiming();
      keys = keys_gen();
      state.ResumeTiming();
      for (size_t j = 0; j != kRepetitions; ++j) {
        auto& s = empty_set_getter();
        for (uint32_t key : keys) {
          auto res = s.insert(key);
          DoNotOptimize(res);
        }
      }
    }
  }
}

// Measures the time it takes to `clear` a set and then `insert` the same number
// of random elements back. The reported time is per element. In other
// words, the pseudo code below counts as N iterations.
//
//   set.erase(set.begin(), set.end());
//   set.insert(key1);
//   ...
//   set.insert(keyN);
//
// What we really need is to clear the container without releasing memory. For
// most containers this can be expressed as `set.clear()` but for SwissTable
// containers this call can release memory.
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertManyUnordered_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  auto gen_set = [&]() {
    return GenerateSet<Set>(state.range(0),
                            static_cast<Density>(state.range(1)));
  };
  Set s;
  std::vector<uint32_t> keys;

  RunInsertManyUnordered_Hot(
      state,
      [&]() {
        // The keys are regenerated every kRepetitions in
        // RunInsertManyUnordered_Hot.
        // Regeneration is important to reduce variance due to specific hash
        // collision pattern.
        keys = ToVector(gen_set());
        // keys vector has order of SwissTable returned by GenerateSet.
        std::shuffle(keys.begin(), keys.end(), absl::BitGen());
        return keys;
      },
      [&]() -> Set& {
        // This guarantee to not release memory that is important for hot
        // benchamrk.
        s.erase(s.begin(), s.end());
        return s;
      });
}

// Measures the time it takes to `insert` elements to empty set.
// The reported time is per element.
// In other words, the pseudo code below counts as N iterations.
//
//   Set set;
//   set.insert(key1);
//   ...
//   set.insert(keyN);
template <template <class...> class SetT, size_t kValueSizeT>
static void BM_SWISSMAP_InsertManyToEmpty_Hot(benchmark::State& state) {
  using Set = SetT<Value<kValueSizeT>, Hash, Eq>;

  const size_t num_keys = state.range(0);

  Set s;
  RunInsertManyUnordered_Hot(
      state,
      [num_keys]() {
        // Generates `num_keys` unique keys.
        std::vector<uint32_t> keys;
        keys.reserve(num_keys);
        for (Set s; keys.size() < num_keys;) {
          uint32_t elem = RandomNonSpecial();
          if (s.insert(elem).second) {
            keys.emplace_back(elem);
          }
        }
        return keys;
      },
      [&s]() -> Set& {
        s = Set();
        return s;
      });
}

using IntTable = absl::flat_hash_set<int64_t>;
using StrTable = absl::flat_hash_set<std::string>;

void BM_SWISSMAP_EmptyConstructor(benchmark::State& state) {
  for (auto unused : state) {
    IntTable t;
    benchmark::DoNotOptimize(t);
  }
}

void BM_SWISSMAP_SizedConstructor(benchmark::State& state) {
  constexpr int kElements = 64;
  for (auto unused : state) {
    IntTable t(kElements);
    benchmark::DoNotOptimize(t);
  }
}

template <class T>
class CustomAlloc : public std::allocator<T> {
 public:
  bool unused_ = true;  // Force it to not look like std::allocator.

  // Default constructor
  CustomAlloc() noexcept = default;

  // Copy constructor
  template <class U>
  explicit CustomAlloc(const CustomAlloc<U>&) noexcept {}

  // Add the rebind mechanism for the allocator to ensure the custom allocators
  // with the correct value type can be used
  template <class U>
  struct rebind {
    typedef CustomAlloc<U> other;
  };
};

void BM_SWISSMAP_MoveConstructor(benchmark::State& state) {
  // For now just measure a small cheap hash table since we
  // are mostly interested in the overhead of type-erasure
  // in resize(). We also use a custom allocator to disable
  // leaking hashtable entries into /hashtablez since we
  // do not destroy hash tables.
  constexpr int kElements = 64;

  using CheapTable =
      absl::flat_hash_set<int64_t, IntTable::hasher, IntTable::key_equal,
                          CustomAlloc<int64_t>>;

  // We swap back and forth between two slots, exactly one of which
  // holds an CheapTable at any point.
  union Space {
    bool not_used;
    CheapTable t;

    Space() { not_used = true; }
    ~Space() {}
  };
  Space space[2];
  int current = 0;
  new (&space[current].t) CheapTable();
  for (int i = 0; i < kElements; i++) {
    space[current].t.insert(i);
  }

  for (auto unused : state) {
    // Move from current to the other slot.
    const int other = 1 - current;
    new (&space[other].t) CheapTable(std::move(space[current].t));
    current = other;
  }

  space[current].t.CheapTable::~CheapTable();
}

ABSL_ATTRIBUTE_NOINLINE void FillInts(IntTable* t, int n) {
  for (int i = 0; i < n; i++) {
    t->insert(i);
  }
}

void BM_SWISSMAP_IntDestructor(benchmark::State& state) {
  int size = state.range(0);
  int capacity = state.range(1);
  size_t batch_size = (capacity <= 1) ? 8196 : 512;
  while (state.KeepRunningBatch(batch_size)) {
    state.PauseTiming();
    std::vector<IntTable> tables(batch_size);
    for (IntTable& t : tables) {
      t.reserve(capacity);
      FillInts(&t, size);
    }
    state.ResumeTiming();
    benchmark::DoNotOptimize(tables);
  }
}

ABSL_ATTRIBUTE_NOINLINE void FillStrings(StrTable* t, int n) {
  assert(n < 256);
  std::string s;
  for (int i = 0; i < n; i++) {
    s.clear();
    s.push_back(static_cast<char>(i));
    t->insert(s);
  }
}

void BM_SWISSMAP_StrDestructor(benchmark::State& state) {
  int size = state.range(0);
  int capacity = state.range(1);
  size_t batch_size = (capacity == 0) ? 16384 : (capacity <= 7) ? 512 : 128;
  while (state.KeepRunningBatch(batch_size)) {
    state.PauseTiming();
    std::vector<StrTable> tables(batch_size);
    for (StrTable& t : tables) {
      t.reserve(capacity);
      FillStrings(&t, size);
    }
    state.ResumeTiming();
    benchmark::DoNotOptimize(tables);
  }
}

void RegisterHotBenchmarks() {
  std::vector<benchmark::internal::Benchmark*> benchmarks;
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindMiss_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindMiss_Hot, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertMiss_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertMiss_Hot, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindHit_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_FindHit_Hot, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertHit_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertHit_Hot, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_Iterate_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_Iterate_Hot, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertManyOrdered_Hot,
                                  4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks, BM_SWISSMAP_InsertManyOrdered_Hot,
                                  64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks,
                                  BM_SWISSMAP_InsertManyUnordered_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(benchmarks,
                                  BM_SWISSMAP_InsertManyUnordered_Hot, 64);
  for (auto* benchmark : benchmarks) {
    benchmark->ArgNames({"set_size", "density"});
    // If UseExplicitIterationCounts() is false, then the code below is
    // equivalent to:
    //     benchmark->RangeMultiplier(2)->Ranges({
    //         {1, 1 << 20},
    //         {static_cast<int64_t>(Density::kMin),
    //          static_cast<int64_t>(Density::kMax)},
    //     });
    // We cannot use the same approach if UseExplicitIterationCounts() is true
    // because it is not possible to set different iteration counts for
    // benchmarks that are part of the same family. We therefore manually do
    // what `Ranges()` does, and register a separate benchmark for the special
    // case for which we want to set an explicit iteration count.
    for (int64_t set_size = 1; set_size <= (1 << 20); set_size *= 2) {
      for (int64_t density = static_cast<int64_t>(Density::kMin);
           density <= static_cast<int64_t>(Density::kMax); density++) {
        if (UseExplicitIterationCounts() &&
            absl::string_view(benchmark->GetName()) ==
                "BM_SWISSMAP_InsertHit_Hot<::absl::flat_hash_set, 64>" &&
            set_size == 64 && density == static_cast<int64_t>(Density::kMin)) {
          REGISTER_BENCHMARK_TEMPLATE(BM_SWISSMAP_InsertHit_Hot,
                                      ::absl::flat_hash_set, 64)
              ->ArgNames({"set_size", "density"})
              ->Args({set_size, density})
              ->Iterations(1);
        } else {
          benchmark->Args({set_size, density});
          benchmark->Iterations(1);
        }
      }
    }
  }

  std::vector<benchmark::internal::Benchmark*> erase_insert_benchmarks;
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(erase_insert_benchmarks,
                                  BM_SWISSMAP_EraseInsert_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(erase_insert_benchmarks,
                                  BM_SWISSMAP_EraseInsert_Hot, 64);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(erase_insert_benchmarks,
                                  BM_SWISSMAP_InsertManyToEmpty_Hot, 4);
  ADD_SWISSMAP_BENCHMARKS_TO_LIST(erase_insert_benchmarks,
                                  BM_SWISSMAP_InsertManyToEmpty_Hot, 64);
  for (auto* benchmark : erase_insert_benchmarks) {
    benchmark->ArgNames({"set_size"})
        ->RangeMultiplier(2)
        ->Ranges({
            {1, 1 << 20},
        });
  }

  REGISTER_BENCHMARK(BM_SWISSMAP_EmptyConstructor);
  REGISTER_BENCHMARK(BM_SWISSMAP_SizedConstructor);
  REGISTER_BENCHMARK(BM_SWISSMAP_MoveConstructor);

  auto destructor_setup_fn = [](auto* b) {
    b->ArgNames({"size", "capacity"})
        // These values are determinated empirically and cover wide variety of
        // capacity and sizes presented in the fleet.
        ->ArgPair(0, 0)
        ->ArgPair(0, 1)
        ->ArgPair(1, 1)
        ->ArgPair(1, 7)
        ->ArgPair(6, 7)
        ->ArgPair(0, 127)
        ->ArgPair(1, 127)
        ->ArgPair(13, 127)
        ->ArgPair(70, 127)
        ->ArgPair(100, 255)
        ->ArgPair(255, 255);
  };

  REGISTER_BENCHMARK(BM_SWISSMAP_IntDestructor)->Apply(destructor_setup_fn);
  REGISTER_BENCHMARK(BM_SWISSMAP_StrDestructor)->Apply(destructor_setup_fn);
}

}  // namespace swissmap

}  // namespace fleetbench
