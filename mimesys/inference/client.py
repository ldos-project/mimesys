"""
mimesys inference client CLI
================================
Interact with a running inference server from the command line.

Usage
-----
  python -m mimesys.inference.client [--url URL] <command> [options]

Commands
--------
  health             Check server liveness
  info               Show model / hardware / config metadata
  metrics            List supported input metrics with training ranges
  generate-series    Generate a time-series of execution plans
  profile-series     Profile a time-series on a remote worker (with live progress)
  profile-result     Fetch result of a previously submitted profile job

Global options
--------------
  --url URL      Server base URL  (default: http://localhost:8000)
  --raw          Print raw JSON response instead of formatted output

Examples
--------
  python -m mimesys.inference.client health

  python -m mimesys.inference.client generate-series \\
      --traces '[{"io":2000},{"avg_cpu_utilizations_core_00":70},{"io":5000}]' \\
      --method diffusion --format h5 --output plan_series.h5

  python -m mimesys.inference.client profile-series \\
      --traces '[{"io":2000},{"avg_cpu_utilizations_core_00":70},{"io":5000}]' \\
      --method diffusion --output metrics.png
"""

from __future__ import annotations

import argparse, json, sys, time
from typing import Optional

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no extra deps)
# ---------------------------------------------------------------------------
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import urllib.request as _urllib_req
    import urllib.error  as _urllib_err
except ImportError:
    pass


def _get(url: str, stream: bool = False):
    if _HAS_REQUESTS:
        r = requests.get(url, stream=stream, timeout=30)
        r.raise_for_status()
        return r
    raise ImportError("pip install requests")


def _post(url: str, body: dict, stream: bool = False):
    if _HAS_REQUESTS:
        r = requests.post(url, json=body, stream=stream, timeout=30)
        r.raise_for_status()
        return r
    raise ImportError("pip install requests")


# ---------------------------------------------------------------------------
# SSE parser (no external dep)
# ---------------------------------------------------------------------------
def _iter_sse(response) -> dict:
    """
    Yield parsed JSON objects from a text/event-stream response.
    Handles multi-line SSE data fields and keepalive comment lines.
    """
    buf = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            for line in block.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
_STAGE_WIDTH = 16
_BAR_WIDTH   = 30

STAGE_COLORS = {
    "queued":           "\033[90m",   # dark grey
    "generating_plan":  "\033[36m",   # cyan
    "packaging":        "\033[36m",
    "connecting":       "\033[33m",   # yellow
    "transferring":     "\033[33m",
    "benchmarking":     "\033[34m",   # blue
    "collecting":       "\033[35m",   # magenta
    "parsing":          "\033[35m",
    "done":             "\033[32m",   # green
    "error":            "\033[31m",   # red
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def _progress_bar(pct: int, width: int = _BAR_WIDTH) -> str:
    filled = int(width * pct / 100)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:3d}%"


def _render_progress(stage: str, pct: int, message: str) -> str:
    color  = STAGE_COLORS.get(stage, "")
    label  = f"{color}{stage:<{_STAGE_WIDTH}}{RESET}"
    bar    = _progress_bar(pct)
    max_msg = 72
    msg = message if len(message) <= max_msg else message[:max_msg - 3] + "..."
    return f"  {label}  {bar}  {msg}"


def _print_progress(stage: str, pct: int, message: str, *, clear: bool = True):
    line = _render_progress(stage, pct, message)
    if clear:
        sys.stdout.write(f"\r{line}\033[K")
    else:
        sys.stdout.write(f"\n{line}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------
def cmd_health(url: str, raw: bool, **_):
    data = _get(f"{url}/health").json()
    if raw:
        _print_json(data); return
    print(f"  status   : {BOLD}{data['status']}{RESET}")
    print(f"  device   : {data['device']}")
    print(f"  epoch    : {data.get('epoch')}")
    print(f"  step     : {data.get('global_step')}")
    print(f"  profiling: {data.get('profiling_enabled')}")
    print(f"  methods  : {', '.join(data.get('methods_available', []))}")


def cmd_info(url: str, raw: bool, **_):
    data = _get(f"{url}/info").json()
    if raw:
        _print_json(data); return
    m = data["model"]
    print(f"  checkpoint   : {m['checkpoint']}")
    print(f"  epoch        : {m['epoch']}  (step {m['global_step']})")
    print(f"  architecture : {m['architecture']}")
    inp = data["input"]
    print(f"  trace_dim    : {inp['trace_dim']}")
    print(f"  prev_action  : {inp['prev_action_shape']}")
    out = data["output"]
    print(f"  action_shape : {out['action_shape']}  →  {out['stressor_names']}")
    hw  = data["training"]["hardware"]
    print(f"  trained on   : {hw['num_machines']}× {hw['machine_type']}  ({hw['cluster']})")
    prof = data["profiling"]
    print(f"  profiling    : {'enabled' if prof['enabled'] else 'disabled'}"
          + (f"  ({len(prof['workers'])} workers)" if prof["enabled"] else ""))


def cmd_metrics(url: str, raw: bool, **_):
    data = _get(f"{url}/metrics").json()
    if raw:
        _print_json(data); return
    print(f"  {'Metric':<42} {'Unit':<8} {'Train min':>12} {'Train max':>12}")
    print("  " + "-" * 78)
    for m in data["metrics"]:
        lo = f"{m['train_min']:.2f}" if m["train_min"] is not None else "?"
        hi = f"{m['train_max']:.2f}" if m["train_max"] is not None else "?"
        print(f"  {m['name']:<42} {m['unit']:<8} {lo:>12} {hi:>12}")


def cmd_generate_series(url: str, raw: bool,
                        traces: str, method: str, format: str,
                        initial_prev_action: Optional[str],
                        n_chains: int, cfg_guide_w: float,
                        output: Optional[str], **_):
    body: dict = {
        "traces":        json.loads(traces),
        "method":        method,
        "return_format": format,
        "n_chains":      n_chains,
        "cfg_guide_w":   cfg_guide_w,
    }
    if initial_prev_action:
        body["initial_prev_action"] = json.loads(initial_prev_action)

    r = _post(f"{url}/generate/series", body)

    if format == "h5":
        path = output or "execution_plan_series.h5"
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  Saved HDF5 series → {path}")
    elif format == "stress_ng":
        print(r.text)
    else:
        data = r.json()
        if raw:
            _print_json(data)
        else:
            print(f"  T={data['T']} steps, method={data['method']}")
            for i, act in enumerate(data["actions"]):
                mx = max(max(row) for row in act)
                print(f"    t={i}: max_load={mx:.3f}")


def cmd_profile_series(url: str, raw: bool,
                       traces: str, method: str,
                       initial_prev_action: Optional[str],
                       n_chains: int, cfg_guide_w: float,
                       output: Optional[str], **_):
    """Submit a time-series profiling job and visualize the per-step metrics."""
    body: dict = {
        "traces":      json.loads(traces),
        "method":      method,
        "n_chains":    n_chains,
        "cfg_guide_w": cfg_guide_w,
    }
    if initial_prev_action:
        body["initial_prev_action"] = json.loads(initial_prev_action)

    resp = _post(f"{url}/profile/series", body)
    job  = resp.json()
    job_id     = job["job_id"]
    stream_url = f"{url}{job['stream_url']}"
    result_url = f"{url}{job['result_url']}"

    print(f"\n  Time-series job submitted: {BOLD}{job_id}{RESET}")
    print(f"  Stream : {stream_url}")
    print(f"  Result : {result_url}")
    print()

    last_stage = None
    try:
        sse_resp = _get(stream_url, stream=True)
        for event in _iter_sse(sse_resp):
            stage   = event.get("stage", "")
            pct     = event.get("pct", 0)
            message = event.get("message", "")

            if stage != last_stage:
                if last_stage is not None:
                    sys.stdout.write("\n")
                last_stage = stage

            _print_progress(stage, pct, message, clear=(stage not in ("done", "error")))

            if stage == "done":
                sys.stdout.write("\n\n")
                sys.stdout.flush()
                break
            if stage == "error":
                sys.stdout.write("\n")
                sys.stdout.flush()
                print(f"\n  {STAGE_COLORS['error']}Error:{RESET} {message}", file=sys.stderr)
                sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n\n  Interrupted — job {job_id} is still running on the server.")
        print(f"  Resume: python -m mimesys.inference.client profile-result --job {job_id}")
        sys.exit(0)

    result = _get(result_url).json()
    if raw:
        _print_json(result); return

    T           = result.get("T", 0)
    series      = result.get("series_metrics", [])
    worker      = result.get("worker", "?")
    method_used = result.get("method", "?")

    print(f"  Worker  : {worker}")
    print(f"  Method  : {method_used}")
    print(f"  Steps   : {T}")
    print()

    if not series:
        print("  (no metrics returned)")
        return

    _print_series_table(series)

    if output:
        _save_series_plot(series, output)
        print(f"\n  Plot saved → {output}")


def cmd_profile_result(url: str, raw: bool, job: str, **_):
    """Fetch the result of a previously submitted profile job."""
    result = _get(f"{url}/profile/jobs/{job}/result").json()
    if raw:
        _print_json(result); return
    if "stage" in result and result.get("stage") != "done":
        print(f"  Status: {result['stage']}  ({result.get('pct', 0)}%)  {result.get('message', '')}")
        return
    print(f"  Worker : {result.get('worker')}")
    print(f"  Method : {result.get('method')}")
    series = result.get("series_metrics", [])
    if series:
        _print_series_table(series)
    else:
        metrics = result.get("measured_metrics", {})
        if metrics:
            _print_series_table([metrics])


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------
def _print_json(data):
    print(json.dumps(data, indent=2))


_METRIC_LABELS = {
    "l3_cache_usage_socket_0":   ("L3 Cache S0",  "MB"),
    "l3_cache_usage_socket_1":   ("L3 Cache S1",  "MB"),
    "memory_bandwidth_socket_0": ("MemBW S0",     "GB/s"),
    "memory_bandwidth_socket_1": ("MemBW S1",     "GB/s"),
    "io":                        ("IO",            "KB/s"),
    "avg_cpu_utilizations_total":("CPU total",    "%"),
}


def _print_series_table(series: list[dict]):
    """Print a per-step table for the key metrics in series_metrics."""
    display_keys = list(_METRIC_LABELS.keys())
    present = [k for k in display_keys if any(k in d for d in series)]
    if not present:
        present = sorted(series[0].keys()) if series else []

    header = f"  {'step':>4}  " + "  ".join(f"{_METRIC_LABELS.get(k, (k,''))[0]:>14}" for k in present)
    units  = f"  {'':>4}  " + "  ".join(f"{'('+_METRIC_LABELS.get(k,('',''))[1]+')':>14}" for k in present)
    sep    = "  " + "-" * (6 + 16 * len(present))
    print(header)
    print(units)
    print(sep)
    for i, d in enumerate(series):
        vals = "  ".join(f"{d.get(k, float('nan')):>14.2f}" for k in present)
        print(f"  {i:>4}  {vals}")


def _save_series_plot(series: list[dict], output_path: str):
    """Save a time-series line plot to output_path (PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    display_keys = [k for k in _METRIC_LABELS if any(k in d for d in series)]
    if not display_keys:
        display_keys = sorted(series[0].keys()) if series else []

    steps = list(range(len(series)))
    n_metrics = len(display_keys)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, key in zip(axes, display_keys):
        label, unit = _METRIC_LABELS.get(key, (key, ""))
        values = [d.get(key, float("nan")) for d in series]
        ax.plot(steps, values, marker="o", linewidth=1.5)
        ax.set_ylabel(f"{label}\n({unit})" if unit else label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, len(steps) - 0.5)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Time-series hardware metrics", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimesys.inference.client",
        description="CLI client for the stress-emulation inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", default="http://localhost:8000",
                   help="Server base URL (default: http://localhost:8000)")
    p.add_argument("--raw", action="store_true",
                   help="Print raw JSON without formatting")

    sub = p.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="Check server liveness")

    # info
    sub.add_parser("info", help="Show model metadata")

    # metrics
    sub.add_parser("metrics", help="List supported input metrics")

    # generate-series
    gs = sub.add_parser("generate-series", help="Generate a time-series of execution plans")
    gs.add_argument("--traces",              required=True,
                    help="JSON list of trace dicts")
    gs.add_argument("--method",              default="diffusion",
                    choices=["diffusion","nearest_neighbor","linear_interpolation","single_stressor"])
    gs.add_argument("--format",              default="json",
                    choices=["h5","json","stress_ng"])
    gs.add_argument("--initial_prev_action", default=None)
    gs.add_argument("--n_chains",            type=int,   default=3)
    gs.add_argument("--cfg_guide_w",         type=float, default=3.0)
    gs.add_argument("--output",              default=None, help="Output path for h5 format")

    # profile-series
    ps = sub.add_parser("profile-series",
                        help="Time-series profiling: generate plans and measure on a remote worker")
    ps.add_argument("--traces",              required=True,
                    help="JSON list of T trace dicts")
    ps.add_argument("--method",              default="diffusion",
                    choices=["diffusion","nearest_neighbor","linear_interpolation","single_stressor"])
    ps.add_argument("--initial_prev_action", default=None,
                    help="JSON (STRESSORS×THREADS) matrix for t=0 prev_action")
    ps.add_argument("--n_chains",            type=int,   default=3)
    ps.add_argument("--cfg_guide_w",         type=float, default=3.0)
    ps.add_argument("--output",              default=None,
                    help="Path to save time-series plot PNG (optional)")

    # profile-result (resume / poll an existing job)
    pres = sub.add_parser("profile-result",
                          help="Fetch result of a previously submitted profile job")
    pres.add_argument("--job", required=True, help="Job ID returned by 'profile-series'")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = _build_parser()
    args   = parser.parse_args()
    kwargs = vars(args)
    cmd    = kwargs.pop("command")

    dispatch = {
        "health":          cmd_health,
        "info":            cmd_info,
        "metrics":         cmd_metrics,
        "generate-series": cmd_generate_series,
        "profile-series":  cmd_profile_series,
        "profile-result":  cmd_profile_result,
    }

    fn = dispatch.get(cmd)
    if fn is None:
        parser.error(f"Unknown command: {cmd}")

    try:
        fn(**kwargs)
    except Exception as e:
        if _HAS_REQUESTS:
            import requests as _req
            if isinstance(e, _req.HTTPError):
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                print(f"\n  {STAGE_COLORS['error']}HTTP {e.response.status_code}:{RESET} {detail}",
                      file=sys.stderr)
                sys.exit(1)
        print(f"\n  {STAGE_COLORS['error']}Error:{RESET} {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
