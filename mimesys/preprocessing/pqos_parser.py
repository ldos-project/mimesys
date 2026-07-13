"""Parser for pqos.log files emitted by mimesys_benchmark's StartProfilingPqos.

Format: TIME-blocked stream, one block per snapshot (libpqos `pqos_mon_poll`):

  TIME 2026-06-20 11:46:05
      CORE         IPC      MISSES     LLC[KB]   MBL[MB/s]   MBR[MB/s]
      0-19        0.77        223k     10120.0        23.0         1.5

Returns dict[group_label] -> dict[metric_name] -> list of per-snapshot floats.

All cores are monitored as one merged group: a per-socket split inflates
variance via workload imbalance, MBL is redundant with hpc memory_bandwidth,
and MBR is meaningless for a group spanning both sockets. Useful fields:
LLC occupancy (KB), IPC, raw L3 misses.
"""
import re
from typing import Dict, List

_ROW_RE = re.compile(
    r"^\s*(?P<core>\d+(?:-\d+)?)"
    r"\s+(?P<ipc>[\d.]+)"
    r"\s+(?P<misses>[\d.]+)(?P<unit>[kKMG]?)"
    r"\s+(?P<llc_kb>[\d.]+)"
    r"\s+(?P<mbl>[\d.]+)"
    r"\s+(?P<mbr>[\d.]+)\s*$"
)
_UNIT = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}


def parse_pqos_log(path: str) -> Dict[str, Dict[str, List[float]]]:
    """Parse pqos.log file. Returns {group_label: {metric: [float, ...]}}.

    Metrics per group: ipc, misses (per-snapshot count), llc_kb, mbl, mbr.
    One value per TIME block. Groups present depend on the pqos -m argument
    used at capture time (merged "0-19", or "0-9" + "10-19").
    """
    groups: Dict[str, Dict[str, List[float]]] = {}
    cur_block: Dict[str, dict] = {}
    try:
        f = open(path)
    except OSError:
        return groups
    with f:
        for line in f:
            line = line.rstrip()
            if line.startswith("TIME "):
                if cur_block:
                    for g, vals in cur_block.items():
                        d = groups.setdefault(g, {
                            "ipc": [], "misses": [],
                            "llc_kb": [], "mbl": [], "mbr": []})
                        for k in ("ipc", "misses", "llc_kb", "mbl", "mbr"):
                            d[k].append(vals[k])
                cur_block = {}
                continue
            m = _ROW_RE.match(line)
            if not m:
                continue
            g = m.group("core")
            cur_block[g] = {
                "ipc": float(m.group("ipc")),
                "misses": float(m.group("misses")) * _UNIT[m.group("unit")],
                "llc_kb": float(m.group("llc_kb")),
                "mbl": float(m.group("mbl")),
                "mbr": float(m.group("mbr")),
            }
    if cur_block:
        for g, vals in cur_block.items():
            d = groups.setdefault(g, {
                "ipc": [], "misses": [],
                "llc_kb": [], "mbl": [], "mbr": []})
            for k in ("ipc", "misses", "llc_kb", "mbl", "mbr"):
                d[k].append(vals[k])
    return groups


# Canonical flat metric names for downstream training (MBL/MBR dropped; see
# module docstring).
PQOS_METRIC_KEYS = [
    "pqos_llc_kb",   # LLC occupancy (KB resident in L3)
    "pqos_ipc",      # instructions per cycle
    "pqos_misses",   # raw L3 miss count / sec
]


def pqos_metrics_dict(path: str,
                      merged_label: str = "0-19") -> Dict[str, List[float]]:
    """Read pqos.log and return a flat dict keyed by PQOS_METRIC_KEYS.

    One TIME block is produced per slot, so sample count matches
    hpcperfstatsd 1:1 with no timestamp alignment needed.

    Uses the merged group ("0-19") when present; otherwise falls back to
    2-group captures ("0-9" + "10-19"), summing llc/misses and averaging IPC.
    """
    g = parse_pqos_log(path)
    d = g.get(merged_label)
    if d is not None:
        return {
            "pqos_llc_kb": d.get("llc_kb", []),
            "pqos_ipc":    d.get("ipc", []),
            "pqos_misses": d.get("misses", []),
        }
    # 2-group fallback.
    s0 = g.get("0-9", {})
    s1 = g.get("10-19", {})
    if not s0 and not s1:
        return {k: [] for k in PQOS_METRIC_KEYS}
    def _sum(k):
        a, b = s0.get(k, []), s1.get(k, [])
        n = min(len(a), len(b)) if a and b else max(len(a), len(b))
        if not a: return list(b)[:n]
        if not b: return list(a)[:n]
        return [a[i] + b[i] for i in range(n)]
    def _mean(k):
        a, b = s0.get(k, []), s1.get(k, [])
        n = min(len(a), len(b)) if a and b else max(len(a), len(b))
        if not a: return list(b)[:n]
        if not b: return list(a)[:n]
        return [(a[i] + b[i]) / 2 for i in range(n)]
    return {
        "pqos_llc_kb": _sum("llc_kb"),
        "pqos_ipc":    _mean("ipc"),
        "pqos_misses": _sum("misses"),
    }
