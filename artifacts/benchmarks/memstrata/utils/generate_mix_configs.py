#!/usr/bin/env python3
"""Generate workload-mix configs honouring a target category distribution.

For each mix size N in --n-range, emit `--mixes-per-n` lines into
exp_configs/<out>/<N>/config.txt, each formatted as

    "workload1 workload2 ..." "vm_id1 vm_id2 ..."

so they can be fed straight into `./mimebench.py run-batch` (the existing
loader in load_config_dir() splits on `" "`).

Distribution: pick each workload's category by weighted sample, with weights
proportional to how far each category is from its target share of the total
workload count generated this run. Across the whole emission this converges
to the requested percentages; per-mix composition is otherwise unconstrained
(modulo the VM-pool capacity rule below).

VM assignment: each workload's VM is sampled (without replacement within the
mix) from VM_POOL_<TYPE>. If a category's pool is too small to host the
workloads already picked for this mix, sampling backtracks and tries a
different category. Categories whose pools are exhausted for the current mix
are excluded from the weighted sample for the remainder of that mix.

Defaults match the paper's Table 1.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "workloads.tsv"
HOST_ENV = ROOT / "config" / "host.env"


# Target distribution (Table 1 in the paper). Category names match the
# `type` column in workloads.tsv (note: "Web Serving" -> webserver).
DEFAULT_TARGETS = {
    "webserver": 0.31,
    "bigdata":   0.32,
    "kvstore":   0.14,
    "ml":        0.11,
    "database":  0.07,
    "graph":     0.05,
}

# Table 1 only names resnet50 + stable_diffusion under ML inference; the
# workload registry's ml type also covers dlrm_*, which we exclude here.
ML_ALLOWED = {"resnet50", "stable_diffusion"}


def load_workloads_by_category() -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for line in REGISTRY.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, wtype, _ = parts
        if wtype == "ml" and name not in ML_ALLOWED:
            continue
        if wtype in DEFAULT_TARGETS:
            by_cat[wtype].append(name)
    return dict(by_cat)


def load_vm_pools() -> dict[str, list[int]]:
    """Parse VM_POOL_<TYPE>:= lines out of config/host.env (env wins)."""
    pools: dict[str, list[int]] = {}
    pat = re.compile(r'^:\s*"\${(VM_POOL_[A-Z]+):=([^}]*)}"')
    for line in HOST_ENV.read_text().splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        pools[key] = [int(x) for x in raw.split(",") if x.strip()]
    out: dict[str, list[int]] = {}
    for cat in DEFAULT_TARGETS:
        env_key = f"VM_POOL_{cat.upper()}"
        raw = os.environ.get(env_key)
        vms = ([int(x) for x in raw.split(",") if x.strip()] if raw
               else pools.get(env_key, []))
        if not vms:
            raise SystemExit(f"VM_POOL_{cat.upper()} is empty in host.env / env")
        out[cat] = vms
    return out


def weighted_pick(weights: dict[str, float]) -> str:
    """Pick a key with probability proportional to its (non-negative) weight."""
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        # Fall back to uniform over candidates with any weight defined.
        return random.choice(list(weights.keys()))
    r = random.random() * total
    acc = 0.0
    for k, w in weights.items():
        acc += max(0.0, w)
        if r <= acc:
            return k
    return next(iter(weights))


def generate_mixes(
    n_range: range,
    mixes_per_n: int,
    targets: dict[str, float],
    workloads_by_cat: dict[str, list[str]],
    vm_pools: dict[str, list[int]],
    seed: int | None,
) -> dict[int, list[tuple[list[str], list[int]]]]:
    if seed is not None:
        random.seed(seed)

    # Compute integer quota per category from the target percentages and the
    # total workload slot count, then nudge to make the totals add up.
    total_slots = mixes_per_n * sum(n_range)
    quotas = {c: int(round(p * total_slots)) for c, p in targets.items()}
    diff = total_slots - sum(quotas.values())
    if diff:
        # Distribute residual among the categories with the largest fractional
        # part (positive) or smallest (negative), to keep proportions stable.
        residuals = sorted(
            ((p * total_slots - quotas[c], c) for c, p in targets.items()),
            reverse=(diff > 0),
        )
        for i in range(abs(diff)):
            _, c = residuals[i % len(residuals)]
            quotas[c] += 1 if diff > 0 else -1

    issued = Counter()
    out: dict[int, list[tuple[list[str], list[int]]]] = {}

    for n in n_range:
        mixes: list[tuple[list[str], list[int]]] = []
        for _ in range(mixes_per_n):
            wl_pick: list[str] = []
            vm_pick: list[int] = []
            used_vms_per_cat: dict[str, set[int]] = defaultdict(set)
            for _ in range(n):
                # Eligible categories: have quota remaining and pool capacity left.
                eligible = {}
                for c, q in quotas.items():
                    remaining = q - issued[c]
                    if remaining <= 0:
                        continue
                    if len(used_vms_per_cat[c]) >= len(vm_pools[c]):
                        continue
                    eligible[c] = remaining
                if not eligible:
                    # Quota allocation can leave a single mix slot without a
                    # category that still has both quota and VM-pool room.
                    # Allow oversampling: relax the quota constraint, keep
                    # the VM-pool constraint.
                    for c in targets:
                        if len(used_vms_per_cat[c]) < len(vm_pools[c]):
                            eligible[c] = 1
                if not eligible:
                    raise RuntimeError(
                        f"Cannot place workload {len(wl_pick)+1}/{n}: all VM pools full"
                    )

                cat = weighted_pick(eligible)
                wl = random.choice(workloads_by_cat[cat])
                vm = random.choice([v for v in vm_pools[cat]
                                    if v not in used_vms_per_cat[cat]])
                wl_pick.append(wl)
                vm_pick.append(vm)
                used_vms_per_cat[cat].add(vm)
                issued[cat] += 1

            mixes.append((wl_pick, vm_pick))
        out[n] = mixes

    # Summary on stderr so the caller can sanity-check distribution.
    print("Category quota / issued / target%:", file=sys.stderr)
    for c in targets:
        print(f"  {c:10s} quota={quotas[c]:4d}  issued={issued[c]:4d}  "
              f"target={targets[c]*100:.1f}%  actual={issued[c]/total_slots*100:.1f}%",
              file=sys.stderr)
    return out


def write_configs(mixes: dict[int, list[tuple[list[str], list[int]]]],
                  out_root: Path) -> None:
    for n, lines in mixes.items():
        d = out_root / str(n)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "config.txt").open("w") as f:
            for wl, vm in lines:
                f.write(f'"{" ".join(wl)}" "{" ".join(str(v) for v in vm)}"\n')


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="exp_configs/generated",
                   help="output root (default: exp_configs/generated)")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=6)
    p.add_argument("--mixes-per-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=None,
                   help="seed RNG for reproducibility")
    args = p.parse_args(argv)

    workloads_by_cat = load_workloads_by_category()
    missing = [c for c in DEFAULT_TARGETS if not workloads_by_cat.get(c)]
    if missing:
        raise SystemExit(f"No workloads registered for: {missing}")
    vm_pools = load_vm_pools()

    mixes = generate_mixes(
        n_range=range(args.n_min, args.n_max + 1),
        mixes_per_n=args.mixes_per_n,
        targets=DEFAULT_TARGETS,
        workloads_by_cat=workloads_by_cat,
        vm_pools=vm_pools,
        seed=args.seed,
    )
    out_root = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    write_configs(mixes, out_root)
    print(f"wrote {sum(len(v) for v in mixes.values())} mixes under {out_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
