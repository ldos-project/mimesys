load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("//fleetbench:utils.bzl", "if_clang")

package(default_visibility = ["//visibility:private"])

## Compression Benchmark
cc_library(
    name = "algorithms",
    srcs = ["compression/algorithms.cc"],
    hdrs = ["compression/algorithms.h"],
    deps = [
        ":algorithms_interface",
        ":zlibwrapper",
        "@ZSTD//:zstdlib",
        "@brotli//:brotlidec",
        "@brotli//:brotlienc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@snappy",
    ],
)

cc_library(
    name = "algorithms_interface",
    hdrs = [
        "compression/algorithms.h",
    ],
    deps = [
        "@ZSTD//:zstdlib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "zlibwrapper",
    srcs = [
        "compression/gzipheader.cc",
        "compression/zlibwrapper.cc",
    ],
    hdrs = [
        "compression/gzipheader.h",
        "compression/zlibwrapper.h",
    ],
    deps = [
        "@chromium_zlib//:zlib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "compression_parameter",
    srcs = ["compression/compression_parameter.cc"],
    hdrs = ["compression/compression_parameter.h"],
    data = glob(["compression_parameters/**"]),
    deps = [
        "//fleetbench/common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

BENCHMARK_NUMS = 100

JSON_FILES = [json.split("/")[-1] for json in glob(["distributions/*"])]

[genrule(
    name = "generate_data_%s" % json,
    srcs = ["distributions/%s" % json],
    outs = ["corpora/%s/corpus_%d" % (
        json.removesuffix(".json"),
        n,
    ) for n in range(BENCHMARK_NUMS)],
    cmd = "$(location //fleetbench/compression/generate_corpora:generate_corpora) --distribution_name=$< --output_dir=$(RULEDIR)/corpora --benchmark_nums=100",
    tools = [
        "//fleetbench/compression/generate_corpora",
    ],
) for json in JSON_FILES]

filegroup(
    name = "generate_data",
    srcs = ["generate_data_%s" % json for json in JSON_FILES],
)

genrule(
    name = "corpus",
    srcs = [":generate_data"],
    outs = ["corpus.zip"],
    cmd = "zip -q -r $(OUTS) $(SRCS)",
    visibility = ["//fleetbench:__pkg__"],
)

## LIBC Benchmark

cc_library(
    name = "utils",
    srcs = ["libc/utils.cc"],
    hdrs = ["libc/utils.h"],
    deps = ["@com_google_absl//absl/log:check"],
)


## proto

NUM_MESSAGES = 10

[proto_library(
    name = "Message%d_proto" % id,
    testonly = True,
    srcs = ["proto/Message%d.proto" % id],
) for id in range(0, NUM_MESSAGES)]

cc_library(
    name = "receiver",
    testonly = True,
    hdrs = ["proto/receiver.h"],
    deps = [
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_benchmark//:benchmark",
    ],
)

[cc_proto_library(
    name = "Message%d_cc_proto" % id,
    testonly = True,
    deps = [":Message%d_proto" % id],
) for id in range(0, NUM_MESSAGES)]

[cc_library(
    name = "access_message_%d" % id,
    testonly = True,
    srcs = ["proto/access_message%d.cc" % id],
    hdrs = [
        "proto/access_message%d.h" % id,
    ],
    deps = [
        ":Message%d_cc_proto" % id,
        ":receiver",
        "@com_google_absl//absl/log:check",
    ],
) for id in range(0, NUM_MESSAGES)]


## Stress Emulate Benchmark
cc_library(
    name = "mimesys_lib",
    testonly = True,
    srcs = [
        "mimesys_benchmark.cc",
        "compression/compression_benchmark.cc",
        "hashing/hashing_benchmark.cc",
        "libc/mem_benchmark.cc",
        "proto/benchmark.cc",
        "proto/lifecycle.cc",
        "stl/generated_cord_benchmarks.cc",
        "stl/cord_benchmark.cc",
        "swissmap/swissmap_benchmark.cc",
    ],
    hdrs = [
        "compression/compression_benchmark.h",
        "hashing/hashing_benchmark.h",
        "libc/mem_benchmark.h",
        "proto/benchmark.h",
        "proto/lifecycle.h",
        "stl/generated_cord_benchmarks.h",
        "stl/cord_benchmark.h",
        "swissmap/swissmap_benchmark.h",
    ],
    visibility = ["//fleetbench:__pkg__"],
    linkopts = ["-lpapi"],
    data = glob(["hashing_distributions/**", "libc_distributions/**", "execution_plans/**", "memstrata_commands/**"]),
    deps = [
        "//fleetbench:dynamic_registrar",
        "//fleetbench/common",
        "//fleetbench/mimesys/simd:simd_library",
        "//fleetbench/mimesys/stress_ng:stress_ng_benchmark_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_benchmark//:benchmark",

        # Compression libraries
        ":algorithms",
        ":compression_parameter",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@libzip//:zip",

        # Hashing libraries
        "@com_google_absl//absl/crc:crc32c",
        "@com_google_absl//absl/hash",

        # LIBC libraries
        ":utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/random:distributions",

        # Proto libraries
        ":receiver",
        "//fleetbench:productivity_reporter",
        "@com_google_absl//absl/strings:cord",
        "@com_google_protobuf//:protobuf",

        # STL libraries
        # Swissmap libraries
    ] + if_clang([
        # SIMD libraries
        "@com_google_research_scann//scann/hashes/internal:lut16_args",
        "@com_google_research_scann//scann/hashes/internal:lut16_interface",
        "@com_google_research_scann//scann/utils:alignment",
        "@com_google_research_scann//scann/utils:common",
        "@com_google_research_scann//scann/utils:fast_top_neighbors",
        "@com_google_research_scann//scann/utils:threads",
        "@com_google_research_scann//scann/utils/intrinsics:flags",
        "@com_google_absl//absl/types:span",
    ]) + [
        ":Message%d_cc_proto" % id
        for id in range(0, NUM_MESSAGES)
    ] + [
        ":access_message_%d" % id
        for id in range(0, NUM_MESSAGES)
    ],
    alwayslink = True,
)


cc_test(
    name = "mimesys_benchmark",
    linkstatic = 1,
    malloc = "@com_google_tcmalloc//tcmalloc",
    tags = ["manual"],
    data = [":corpus"],
    deps = [
        ":mimesys_lib",
        "//fleetbench:benchmark_main",
    ],
)
