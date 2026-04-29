TIMESTAMP_UNIT_MS = 100
METRIC_INTERVAL_MS = 500
METRIC_INTERVAL_TO_US = 1e3 * METRIC_INTERVAL_MS

max_time_steps = 120
num_metrics_to_collect = 1 # int(max_time_steps // (METRIC_INTERVAL_MS / TIMESTAMP_UNIT_MS))

# Hardware specs keyed by machine name.
system_specs = {
    "Xeon 8280": {
        "CPU": "56",
        "clock": "2.7",
        "Threads": "112",
        "L1 Cache": "32",    # KB I + 32 KB D per core
        "L2 Cache": "1024",  # KB per core
        "L3 Cache": "38.5",  # MB per socket (77 MB/node)
        "Memory": "192",     # GB DDR4-2933
        "Memory BW": "281.4",  # GB/s
        "Network": "100",    # Gbps Mellanox HDR-100
    },
    "Xeon E5-2620": {
        "CPU": "16",
        "clock": "2.1",     # GHz
        "Threads": "32",
        "L1 Cache": "32",
        "L2 Cache": "256",
        "L3 Cache": "20",   # MB per socket (40 MB/node)
        "Memory": "128",    # GB DDR4-2400
        "Memory BW": "68.3",
        "Network": "56",    # Mellanox FDR (56 Gbps, ~7 GB/s)
    },
    "c220g2": {
        "CPU": "20",
        "clock": "2.6",     # GHz
        "Threads": "40",
        "L1 Cache": "64",
        "L2 Cache": "256",
        "L3 Cache": "20",   # MB per socket (25 MB/node)
        "Memory": "160",    # GB DDR4-2400
        "Memory BW": "59",
        "Network": "10",
    },
}

machine_types = list(system_specs.keys())

# Look up machine spec by logical CPU count (includes hyperthreading).
system_by_num_cores = {
    16:  system_specs["Xeon E5-2620"],
    32:  system_specs["Xeon E5-2620"],
    56:  system_specs["Xeon 8280"],
    112: system_specs["Xeon 8280"],
    20:  system_specs["c220g2"],
    40:  system_specs["c220g2"],
}
