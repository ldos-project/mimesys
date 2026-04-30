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
  profile-from-file  Profile using a TACC stats trace file as input
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

  python -m mimesys.inference.client profile-from-file \\
      --file /path/to/stats-workload.txt \\
      --method diffusion --output metrics.png
"""

from __future__ import annotations

import time, os
import argparse, json, sys
from typing import Optional
from mimesys.preprocessing.parsers import parse_trace_file, process_trace_fine_grained

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no extra deps)
# ---------------------------------------------------------------------------
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def _get(url: str, stream: bool = False):
    if _HAS_REQUESTS:
        # For SSE streams use (connect=10s, read=None) so we never time out
        # waiting for the next event.  For regular GETs, 30s total is fine.
        timeout = (10, None) if stream else 30
        r = requests.get(url, stream=stream, timeout=timeout)
        r.raise_for_status()
        return r
    raise ImportError("pip install requests")


def _post(url: str, body: dict, timeout: int = 360):
    if _HAS_REQUESTS:
        r = requests.post(url, json=body, timeout=timeout)
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
    print(f"  trained on   : {hw['num_machines']}x {hw['machine_type']}  ({hw['cluster']})")
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
                        generation_strategy: Optional[str],
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
    if generation_strategy:
        body["generation_strategy"] = generation_strategy

    r = _post(f"{url}/generate/series", body)

    if format == "h5":
        path = output or "execution_plan_series.h5"
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  Saved HDF5 series → {path}")
    else:
        data = r.json()
        if raw:
            _print_json(data)
        else:
            print(f"  T={data['T']} steps, method={data['method']}")
            for i, act in enumerate(data["actions"]):
                mx = max(max(row) for row in act)
                print(f"    t={i}: max_load={mx:.3f}")


def _parse_trace_file_to_steps(file_path: str) -> list[dict]:
    """Parse a TACC stats file into a list of per-time-step metric dicts.

    Uses the same pipeline as the dataloader: parse_trace_file →
    process_trace_fine_grained (period=2, no fixed duration).  Returns one
    dict per time step, where each dict maps metric names to raw float values.
    """

    _, parsed_traces = parse_trace_file(file_path)
    if not parsed_traces:
        print(f"  No traces found in {file_path}", file=sys.stderr)
        sys.exit(1)

    metrics = process_trace_fine_grained(parsed_traces, period=2, duration=-1)
    if not metrics:
        print(f"  process_trace_fine_grained returned nothing for {file_path}", file=sys.stderr)
        sys.exit(1)

    T = len(next(iter(metrics.values())))
    return [{k: v[t] for k, v in metrics.items()} for t in range(T)]


def _submit_and_stream_profile(url: str, body: dict, raw: bool, output: Optional[str]):
    """POST to /profile/series, stream SSE progress, then print/plot results."""
    input_traces: list[dict] = body.get("traces", [])

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

            if stage == "benchmarking":
                if "duration" in message:
                    message = f"Duration {message.split('duration:')[-1].strip()}"
                    _print_progress(stage, pct, message, clear=(stage not in ("done", "error")))
            else:
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

    series       = _enrich_with_cpu_total(series)
    input_traces = _enrich_with_cpu_total(input_traces)

    # Fetch training ranges once — used for both error normalization and plot ylim.
    metric_ranges: dict = {}
    try:
        ranges_data   = _get(f"{url}/metrics").json()
        metric_ranges = {m["name"]: (m["train_min"], m["train_max"])
                         for m in ranges_data["metrics"]}
    except Exception:
        pass

    _print_series_table(series)

    errors: dict = {}
    if input_traces:
        errors = _compute_error_metrics(series, input_traces, metric_ranges)
        _print_error_table(errors)

    if output:
        _save_series_plot(series, output,
                          input_traces=input_traces or None,
                          metric_ranges=metric_ranges,
                          errors=errors)
        print(f"\n  Plot saved → {output}")


def cmd_profile_series(url: str, raw: bool,
                       traces: str, method: str,
                       initial_prev_action: Optional[str],
                       n_chains: int, cfg_guide_w: float,
                       generation_strategy: Optional[str],
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
    if generation_strategy:
        body["generation_strategy"] = generation_strategy
    _submit_and_stream_profile(url, body, raw, output)


def cmd_generate_from_file(url: str, raw: bool,
                           file: str, method: str,
                           initial_prev_action: Optional[str],
                           n_chains: int, cfg_guide_w: float,
                           generation_strategy: Optional[str],
                           output: Optional[str], **_):
    """Parse a TACC stats trace file and generate an H5 execution plan series."""
    steps = _parse_trace_file_to_steps(file)
    print(f"  Parsed {len(steps)} time steps from {file}")

    body: dict = {
        "traces":        steps,
        "method":        method,
        "return_format": "h5",
        "n_chains":      n_chains,
        "cfg_guide_w":   cfg_guide_w,
    }
    if initial_prev_action:
        body["initial_prev_action"] = json.loads(initial_prev_action)
    if generation_strategy:
        body["generation_strategy"] = generation_strategy

    r = _post(f"{url}/generate/series", body)
    path = output or "execution_plan_series.h5"
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"  Saved HDF5 series ({len(steps)} steps) → {path}")


def cmd_profile_from_file(url: str, raw: bool,
                          file: str, method: str,
                          initial_prev_action: Optional[str],
                          n_chains: int, cfg_guide_w: float,
                          generation_strategy: Optional[str],
                          output: Optional[str], **_):
    """Parse a TACC stats trace file and submit a time-series profiling job."""
    steps = _parse_trace_file_to_steps(file)
    print(f"  Parsed {len(steps)} time steps from {file}")

    body: dict = {
        "traces":      steps,
        "method":      method,
        "n_chains":    n_chains,
        "cfg_guide_w": cfg_guide_w,
    }
    if initial_prev_action:
        body["initial_prev_action"] = json.loads(initial_prev_action)
    if generation_strategy:
        body["generation_strategy"] = generation_strategy
    _submit_and_stream_profile(url, body, raw, output)


def cmd_profile_batch(url: str, raw: bool,
                      files: list[str],
                      method: str,
                      generation_strategy: Optional[str],
                      n_chains: int, cfg_guide_w: float,
                      poll_interval: int,
                      output: Optional[str], **_):
    """Parse a list of TACC stats files, submit as a batch, poll until done, summarize."""

    # ── 1. Parse all files ───────────────────────────────────────────────────
    print(f"\n  Parsing {len(files)} trace file(s)...")
    jobs_input: list[tuple[str, list[dict]]] = []
    for path in files:
        steps = _enrich_with_cpu_total(_parse_trace_file_to_steps(path))
        jobs_input.append((path, steps))
        print(f"    {os.path.basename(path)}: {len(steps)} steps")

    # ── 2. Submit batch ──────────────────────────────────────────────────────
    body: dict = {
        "jobs": [
            {"traces": steps, "method": method,
             **({"generation_strategy": generation_strategy} if generation_strategy else {}),
             "n_chains": n_chains, "cfg_guide_w": cfg_guide_w}
            for _, steps in jobs_input
        ],
        "method": method,
        "n_chains": n_chains,
        "cfg_guide_w": cfg_guide_w,
    }
    if generation_strategy:
        body["generation_strategy"] = generation_strategy

    resp = _post(f"{url}/profile/batch", body)
    batch = resp.json()
    batch_id = batch["batch_id"]
    total    = batch["total"]
    print(f"\n  Batch {BOLD}{batch_id}{RESET} submitted — {total} jobs")
    print(f"  Status: GET {url}/profile/batch/{batch_id}/status\n")

    # ── 3. Poll until all done ───────────────────────────────────────────────
    while True:
        time.sleep(poll_interval)
        status = _get(f"{url}/profile/batch/{batch_id}/status").json()
        batch_stage = status.get("batch_stage", "")
        batch_pct   = status.get("batch_pct", 0)
        batch_msg   = status.get("batch_message", "")
        done    = status["done"]
        error   = status["error"]
        running = status["running"]
        queued  = status["queued"]
        total   = status["total"]
        free    = status.get("workers_free", "?")

        if batch_stage in ("generating_plans", "dispatching", "error"):
            bar_w  = 28
            filled = int(bar_w * batch_pct / 100)
            bar    = "█" * filled + "░" * (bar_w - filled)
            color  = STAGE_COLORS.get(batch_stage, "")
            sys.stdout.write(
                f"\r  {color}{batch_stage:<18}{RESET}  [{bar}] {batch_pct:3d}%  {batch_msg[:60]}\033[K"
            )
        else:
            sys.stdout.write(
                f"\r  {BOLD}{done}/{total}{RESET} done  {error} failed  "
                f"{running} running  {queued} queued  ({free} workers free)\033[K"
            )
        sys.stdout.flush()
        if status["all_done"]:
            break
    sys.stdout.write("\n\n")

    # ── 4. Fetch results for each job ────────────────────────────────────────
    print("  Fetching results...")
    metric_ranges: dict = {}
    try:
        ranges_data  = _get(f"{url}/metrics").json()
        metric_ranges = {m["name"]: (m["train_min"], m["train_max"])
                         for m in ranges_data["metrics"]}
    except Exception:
        pass

    job_ids = list(status.get("job_statuses", {}).keys())
    results: list[dict] = []
    for (fpath, input_steps), job_id in zip(jobs_input, job_ids):
        job_status = status["job_statuses"][job_id]
        if job_status["stage"] == "error":
            print(f"    SKIP {os.path.basename(fpath)} — error: {job_status['message']}")
            continue
        result = _get(f"{url}/profile/jobs/{job_id}/result").json()
        series = _enrich_with_cpu_total(result.get("series_metrics", []))
        if not series:
            print(f"    SKIP {os.path.basename(fpath)} — empty series")
            continue
        errors = _compute_error_metrics(series, input_steps, metric_ranges)
        results.append({
            "file":   fpath,
            "input":  input_steps,
            "output": series,
            "errors": errors,
            "worker": result.get("worker", "?"),
        })

    if not results:
        print("  No successful results to summarize.")
        return

    # ── 5. Print per-file summary ─────────────────────────────────────────────
    print(f"  Results from {len(results)}/{len(jobs_input)} files:\n")
    display_keys = [k for k in _METRIC_LABELS if any(k in r["errors"] for r in results)]
    norm_suffix = " (norm)" if any(
        results[0]["errors"].get(k, {}).get("mse_norm") is not None
        for k in display_keys if k in results[0]["errors"]
    ) else " (raw)"
    header = f"  {'File':<30} " + " ".join(f"{_METRIC_LABELS[k][0]:>14}" for k in display_keys)
    print(header + f"  [MSE{norm_suffix}]")
    print("  " + "-" * (32 + 15 * len(display_keys)))
    for r in results:
        row = f"  {os.path.basename(r['file']):<30} "
        row += " ".join(f"{r['errors'].get(k, {}).get('mse', float('nan')):>14.3f}"
                        for k in display_keys)
        print(row)

    # ── 6. Aggregate summary ─────────────────────────────────────────────────
    import numpy as np
    print(f"\n  Aggregate across {len(results)} files:\n")
    print(f"  {'Metric':<42} {'Mean MSE':>12} {'Std MSE':>10} {'Mean DTW':>12} {'Std DTW':>10}")
    print("  " + "-" * 90)
    agg: dict = {}
    use_norm = any(
        results[0]["errors"].get(k, {}).get("mse_norm") is not None
        for k in display_keys if k in results[0]["errors"]
    )
    mse_key = "mse_norm" if use_norm else "mse"
    dtw_key = "dtw_norm" if use_norm else "dtw"
    suffix  = " (norm)" if use_norm else " (raw)"

    for k in display_keys:
        mse_vals = [r["errors"][k][mse_key] for r in results
                    if k in r["errors"] and r["errors"][k][mse_key] is not None]
        dtw_vals = [r["errors"][k][dtw_key] for r in results
                    if k in r["errors"] and r["errors"][k][dtw_key] is not None]
        agg[k] = {
            "mean_mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
            "std_mse":  float(np.std(mse_vals))  if mse_vals else float("nan"),
            "mean_dtw": float(np.mean(dtw_vals)) if dtw_vals else float("nan"),
            "std_dtw":  float(np.std(dtw_vals))  if dtw_vals else float("nan"),
        }

    print(f"  {'Metric':<42} {f'Mean MSE{suffix}':>18} {'Std':>10} "
          f"{f'Mean DTW{suffix}':>18} {'Std':>10}")
    print("  " + "-" * 102)
    for k in display_keys:
        label = _METRIC_LABELS[k][0]
        a = agg[k]
        print(f"  {label:<42} {a['mean_mse']:>18.6f} {a['std_mse']:>10.6f} "
              f"{a['mean_dtw']:>18.6f} {a['std_dtw']:>10.6f}")

    if raw:
        _print_json({"batch_id": batch_id, "results": len(results), "aggregate": agg})
        return

    # ── 7. Plot avg±std per metric ────────────────────────────────────────────
    if output:
        _save_batch_summary_plot(results, display_keys, agg, metric_ranges, output)
        print(f"\n  Summary plot saved → {output}")


def _save_batch_summary_plot(
    results: list[dict],
    display_keys: list[str],
    agg: dict,
    metric_ranges: dict,
    output_path: str,
):
    """Bar chart: x=metrics, y=mean DTW and MSE (±std) across all batch files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    use_norm = any(
        results[0]["errors"].get(k, {}).get("mse_norm") is not None
        for k in display_keys if k in results[0]["errors"]
    )
    mse_key = "mse_norm" if use_norm else "mse"
    dtw_key = "dtw_norm" if use_norm else "dtw"
    norm_tag = " (norm)" if use_norm else " (raw)"

    labels   = [_METRIC_LABELS.get(k, (k, ""))[0] for k in display_keys]
    mse_mean = [agg[k]["mean_mse"] for k in display_keys]
    mse_std  = [agg[k]["std_mse"]  for k in display_keys]
    dtw_mean = [agg[k]["mean_dtw"] for k in display_keys]
    dtw_std  = [agg[k]["std_dtw"]  for k in display_keys]

    x     = np.arange(len(display_keys))
    width = 0.38

    fig, (ax_mse, ax_dtw) = plt.subplots(2, 1, figsize=(max(8, len(display_keys) * 1.4), 9))

    ax_mse.bar(x, mse_mean, width * 2, yerr=mse_std, capsize=5,
               color="#1f77b4", alpha=0.85, error_kw={"linewidth": 1.5})
    ax_mse.set_ylabel(f"MSE{norm_tag}", fontsize=13)
    ax_mse.set_title(f"MSE{norm_tag} per metric  ({len(results)} files)", fontsize=14)
    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(labels, fontsize=12, rotation=20, ha="right")
    ax_mse.tick_params(axis="y", labelsize=11)
    ax_mse.grid(axis="y", alpha=0.3)
    ax_mse.set_xlim(-0.6, len(x) - 0.4)

    ax_dtw.bar(x, dtw_mean, width * 2, yerr=dtw_std, capsize=5,
               color="#ff7f0e", alpha=0.85, error_kw={"linewidth": 1.5})
    ax_dtw.set_ylabel(f"DTW{norm_tag}", fontsize=13)
    ax_dtw.set_title(f"DTW{norm_tag} per metric  ({len(results)} files)", fontsize=14)
    ax_dtw.set_xticks(x)
    ax_dtw.set_xticklabels(labels, fontsize=12, rotation=20, ha="right")
    ax_dtw.tick_params(axis="y", labelsize=11)
    ax_dtw.grid(axis="y", alpha=0.3)
    ax_dtw.set_xlim(-0.6, len(x) - 0.4)

    fig.suptitle(f"Batch profiling summary ({len(results)} files)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    "l3_cache_usage_socket_0":    ("L3 Cache S0",  "MB/s"),
    "l3_cache_usage_socket_1":    ("L3 Cache S1",  "MB/s"),
    "memory_bandwidth_socket_0":  ("MemBW S0",     "GB/s"),
    "memory_bandwidth_socket_1":  ("MemBW S1",     "GB/s"),
    "io":                         ("IO",            "KB/s"),
    "avg_cpu_utilizations_total": ("CPU total",    "%"),
}


def _avg_cpu_total(d: dict) -> float:
    core_keys = [k for k in d if k.startswith("avg_cpu_utilizations_core_")]
    vals = [d[k] for k in core_keys]
    return sum(vals) / len(vals) if vals else float("nan")


def _enrich_with_cpu_total(steps: list[dict]) -> list[dict]:
    """Add avg_cpu_utilizations_total from per-core keys if not already present."""
    return [
        {**d, "avg_cpu_utilizations_total": _avg_cpu_total(d)}
        if "avg_cpu_utilizations_total" not in d else d
        for d in steps
    ]


def _compute_error_metrics(
    series: list[dict],
    input_traces: list[dict],
    metric_ranges: Optional[dict] = None,
) -> dict:
    """Compute DTW and MSE between output series and input traces per metric.

    Both raw and normalized values are stored.  The normalization factor is
    T × (max − min), where T = min(len(series), len(input_traces)) and
    max/min come from metric_ranges (training-data extremes).  This makes
    errors scale-independent and comparable across metrics.
    """
    import numpy as np
    try:
        from fastdtw import fastdtw
        _has_dtw = True
    except ImportError:
        _has_dtw = False

    common_keys = sorted(set(series[0].keys()) & set(input_traces[0].keys()))
    T_min = min(len(series), len(input_traces))

    errors = {}
    for key in common_keys:
        out_v = np.array([d.get(key, 0.0) for d in series],       dtype=float)
        in_v  = np.array([d.get(key, 0.0) for d in input_traces], dtype=float)
        mse   = float(np.mean((out_v[:T_min] - in_v[:T_min]) ** 2))
        dtw   = float(fastdtw(out_v, in_v, dist=lambda a, b: abs(float(a) - float(b)))[0]) if _has_dtw else None

        # Normalization: divide by T × (max − min).
        # Prefer training-data range; fall back to observed data range for
        # synthetic keys (e.g. avg_cpu_utilizations_total) not in /metrics.
        norm_factor = None
        lo, hi = None, None
        if metric_ranges and key in metric_ranges:
            lo, hi = metric_ranges[key]
        if lo is None or hi is None or (hi - lo) <= 0:
            combined = np.concatenate([out_v, in_v])
            lo, hi   = float(np.nanmin(combined)), float(np.nanmax(combined))
        if (hi - lo) > 0:
            norm_factor = T_min * (hi - lo)

        errors[key] = {
            "mse":      mse,
            "dtw":      dtw,
            "mse_norm": mse / norm_factor       if norm_factor else None,
            "dtw_norm": dtw / norm_factor       if (norm_factor and dtw is not None) else None,
            "norm_factor": norm_factor,
        }
    return errors


def _print_error_table(errors: dict):
    """Print normalized (and raw) DTW and MSE per display metric."""
    display_keys = [k for k in _METRIC_LABELS if k in errors]
    if not display_keys:
        return

    has_dtw  = any(errors[k]["dtw"]      is not None for k in display_keys)
    has_norm = any(errors[k]["mse_norm"] is not None for k in display_keys)

    hdr = f"\n  {'Metric':<30}"
    if has_norm:
        hdr += f"  {'MSE (norm)':>12}  {'DTW (norm)':>12}" if has_dtw else f"  {'MSE (norm)':>12}"
    hdr += f"  {'MSE (raw)':>14}"
    if has_dtw:
        hdr += f"  {'DTW (raw)':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    for k in display_keys:
        e     = errors[k]
        label = _METRIC_LABELS[k][0]
        row   = f"  {label:<30}"
        if has_norm:
            mn = f"{e['mse_norm']:.6f}" if e["mse_norm"] is not None else "      n/a"
            dn = f"{e['dtw_norm']:.6f}" if (has_dtw and e["dtw_norm"] is not None) else "      n/a"
            row += f"  {mn:>12}"
            if has_dtw:
                row += f"  {dn:>12}"
        row += f"  {e['mse']:>14.4f}"
        if has_dtw and e["dtw"] is not None:
            row += f"  {e['dtw']:>14.4f}"
        print(row)


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


def _save_series_plot(
    series: list[dict],
    output_path: str,
    input_traces: Optional[list[dict]] = None,
    metric_ranges: Optional[dict] = None,
    errors: Optional[dict] = None,
):
    """Save a time-series plot overlaying output vs input, with DTW/MSE annotations."""
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

    steps_out = list(range(len(series)))
    steps_in  = list(range(len(input_traces))) if input_traces else []
    n = len(display_keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, display_keys):
        label, unit = _METRIC_LABELS.get(key, (key, ""))

        # Output (profiled)
        out_vals = [d.get(key, float("nan")) for d in series]
        ax.plot(steps_out, out_vals, marker="o", linewidth=1.5,
                color="#1f77b4", label="Output (profiled)")

        # Input (target trace)
        if input_traces:
            in_vals = [d.get(key, float("nan")) for d in input_traces]
            ax.plot(steps_in, in_vals, marker="s", linewidth=1.5,
                    linestyle="--", color="#ff7f0e", label="Input (target)")

        # y-axis ceiling from training data range; CPU total falls back to 100%
        if metric_ranges and key in metric_ranges:
            _, train_max = metric_ranges[key]
            if train_max is not None and train_max > 0:
                ax.set_ylim(bottom=0, top=train_max * 1.05)
        elif key == "avg_cpu_utilizations_total":
            ax.set_ylim(bottom=0, top=100 * 1.05)

        # Subplot title with error metrics (prefer normalized values)
        title = f"{label} ({unit})"
        if errors and key in errors:
            e = errors[key]
            mse_val = e.get("mse_norm") if e.get("mse_norm") is not None else e["mse"]
            dtw_val = e.get("dtw_norm") if e.get("dtw_norm") is not None else e.get("dtw")
            norm_tag = " (norm)" if e.get("mse_norm") is not None else ""
            parts = [f"MSE{norm_tag}={mse_val:.4f}"]
            if dtw_val is not None:
                parts.append(f"DTW{norm_tag}={dtw_val:.4f}")
            title += "   [" + "  ".join(parts) + "]"
        ax.set_title(title, fontsize=13, loc="left")
        ax.set_ylabel(unit, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, max(len(steps_out), len(steps_in)) - 0.5)

    axes[0].legend(fontsize=11, loc="upper right")
    axes[-1].set_xlabel("Time step", fontsize=12)
    axes[-1].tick_params(axis="x", labelsize=11)
    fig.suptitle("Time-series hardware metrics: output vs input", fontsize=15)
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
                    choices=["h5","json"])
    gs.add_argument("--initial_prev_action",   default=None)
    gs.add_argument("--n_chains",              type=int,   default=3)
    gs.add_argument("--cfg_guide_w",           type=float, default=3.0)
    gs.add_argument("--generation_strategy",   default=None,
                    choices=["autoregressive", "parallel_refine"])
    gs.add_argument("--output",                default=None, help="Output path for h5 format")

    # profile-series
    ps = sub.add_parser("profile-series",
                        help="Time-series profiling: generate plans and measure on a remote worker")
    ps.add_argument("--traces",              required=True,
                    help="JSON list of T trace dicts")
    ps.add_argument("--method",              default="diffusion",
                    choices=["diffusion","nearest_neighbor","linear_interpolation","single_stressor"])
    ps.add_argument("--initial_prev_action",   default=None,
                    help="JSON (STRESSORSxTHREADS) matrix for t=0 prev_action")
    ps.add_argument("--n_chains",              type=int,   default=3)
    ps.add_argument("--cfg_guide_w",           type=float, default=3.0)
    ps.add_argument("--generation_strategy",   default=None,
                    choices=["autoregressive", "parallel_refine"])
    ps.add_argument("--output",                default=None,
                    help="Path to save time-series plot PNG (optional)")

    # generate-from-file
    gf = sub.add_parser("generate-from-file",
                        help="Parse a TACC stats trace file and save an H5 execution plan series")
    gf.add_argument("--file",                required=True,
                    help="Path to a TACC stats trace file")
    gf.add_argument("--method",              default="diffusion",
                    choices=["diffusion","nearest_neighbor","linear_interpolation","single_stressor"])
    gf.add_argument("--initial_prev_action",   default=None)
    gf.add_argument("--n_chains",              type=int,   default=3)
    gf.add_argument("--cfg_guide_w",           type=float, default=3.0)
    gf.add_argument("--generation_strategy",   default=None,
                    choices=["autoregressive", "parallel_refine"])
    gf.add_argument("--output",                default=None,
                    help="Output .h5 path (default: execution_plan_series.h5)")

    # profile-from-file
    pf = sub.add_parser("profile-from-file",
                        help="Parse a TACC stats trace file and run time-series profiling")
    pf.add_argument("--file",                required=True,
                    help="Path to a TACC stats trace file (e.g. stats-workload.txt)")
    pf.add_argument("--method",              default="diffusion",
                    choices=["diffusion","nearest_neighbor","linear_interpolation","single_stressor"])
    pf.add_argument("--initial_prev_action",   default=None,
                    help="JSON (STRESSORSxTHREADS) matrix for t=0 prev_action")
    pf.add_argument("--n_chains",              type=int,   default=3)
    pf.add_argument("--cfg_guide_w",           type=float, default=3.0)
    pf.add_argument("--generation_strategy",   default=None,
                    choices=["autoregressive", "parallel_refine"])
    pf.add_argument("--output",                default=None,
                    help="Path to save time-series plot PNG (optional)")

    # profile-batch
    pb = sub.add_parser("profile-batch",
                        help="Submit a batch of trace files, poll until done, print/plot summary")
    pb.add_argument("--files",               required=True, nargs="+",
                    help="One or more TACC stats trace file paths (supports shell globs via quotes)")
    pb.add_argument("--method",              default="diffusion",
                    choices=["diffusion","nearest_neighbor","linear_interpolation","single_stressor"])
    pb.add_argument("--generation_strategy", default=None,
                    choices=["autoregressive", "parallel_refine"])
    pb.add_argument("--n_chains",            type=int,   default=3)
    pb.add_argument("--cfg_guide_w",         type=float, default=3.0)
    pb.add_argument("--poll_interval",       type=int,   default=15,
                    help="Seconds between status polls (default: 15)")
    pb.add_argument("--output",              default=None,
                    help="Path to save summary PNG plot (optional)")

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
        "health":            cmd_health,
        "info":              cmd_info,
        "metrics":           cmd_metrics,
        "generate-series":   cmd_generate_series,
        "generate-from-file": cmd_generate_from_file,
        "profile-series":    cmd_profile_series,
        "profile-from-file": cmd_profile_from_file,
        "profile-batch":     cmd_profile_batch,
        "profile-result":    cmd_profile_result,
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
