"""
FastAPI server for the stress-emulation inference service.

Endpoints
---------
  GET  /health                          - Liveness check
  GET  /info                            - Model / hardware / config metadata
  GET  /metrics                         - Supported input metrics with training-corpus ranges
  POST /generate                        - Single time-step plan from one resource-usage snapshot
  POST /generate/series                 - Long time-series: ordered list of traces → sequence of plans
  POST /profile                         - Submit profiling job; returns job_id immediately
  GET  /profile/jobs/{job_id}/stream    - SSE stream of progress events for a profiling job
  GET  /profile/jobs/{job_id}/result    - Poll for the final result (or current status if pending)
                                          (requires enable_profiling=True)

Profile progress stages
-----------------------
  queued → generating_plan → packaging → connecting → transferring →
  benchmarking (per-batch updates) → collecting → parsing → done | error

Generation methods
------------------
  diffusion            - Pretrained diffusion model (autoregressive prev-action)
  nearest_neighbor     - Closest training-set action by L2 trace distance
  linear_interpolation - Inverse-distance-weighted blend of 5 nearest training actions
  single_stressor      - Heuristic: map per-core CPU utilisation to the cpu stressor
"""

from __future__ import annotations

import asyncio, io, json, os, queue, tempfile, uuid, zipfile
from contextlib import asynccontextmanager
from typing import Literal, Optional

import h5py
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Hydra config
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from mimesys.schema.machine import Machine
from mimesys.preprocessing.dataloader import CustomDataLoader
from mimesys.preprocessing.parsers import parse_trace_file, process_trace_all
from mimesys.inference.engine import (
    InferenceEngine,
    METRIC_KEYS, METRIC_UNITS, STRESSOR_NAMES,
    ACTION_STRESSORS, ACTION_THREADS,
)

# ---------------------------------------------------------------------------
# Server-wide state
# ---------------------------------------------------------------------------
_state: dict = {}

# job_id → {stage, pct, message, result, error, events: asyncio.Queue}
_jobs: dict[str, dict] = {}

# batch_id → {job_ids, total}
_batches: dict[str, dict] = {}

# Thread-safe queue of available worker hostnames.  Populated at startup.
# Each _profile_*_sync call pops one hostname (blocking), returns it when done.
_worker_queue: queue.Queue = queue.Queue()

TRAINING_HARDWARE = {
    "machine_type": "CloudLab c220g2",
    "cpu":          "Intel Xeon E5-2660 v3 (2 sockets x 10 cores, 20 threads)",
    "memory":       "160 GB DDR4",
    "storage":      "480 GB SSD (Intel DC S3500)",
    "num_machines": 14,
    "cluster":      "CloudLab Wisconsin",
}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
METHOD_CHOICES = Literal[
    "diffusion", "nearest_neighbor", "linear_interpolation", "single_stressor"
]
FORMAT_CHOICES = Literal["h5", "json"]
STRATEGY_CHOICES = Literal["autoregressive", "parallel_refine"]


class GenerateRequest(BaseModel):
    trace: dict[str, float] = Field(
        ...,
        description=(
            "Resource-usage snapshot keyed by metric name. Missing keys default to 0. "
            "Values in raw physical units (%, KB/s, MB, GB/s). See GET /metrics."
        ),
        examples=[{"avg_cpu_utilizations_core_00": 72.5, "io": 3200.0,
                   "l3_cache_usage_socket_0": 8500.0, "memory_bandwidth_socket_0": 12.3}],
    )
    method: METHOD_CHOICES = Field(
        "diffusion",
        description=(
            "**diffusion** - pretrained diffusion model (best quality)  |  "
            "**nearest_neighbor** - closest training-set action by L2 trace distance  |  "
            "**linear_interpolation** - IDW blend of 5 nearest neighbours  |  "
            "**single_stressor** - heuristic CPU-only baseline"
        ),
    )
    prev_action: Optional[list[list[float]]] = Field(
        None,
        description=(
            f"Previous action, shape ({ACTION_STRESSORS}, {ACTION_THREADS}), "
            "values in [0, 1]. Used only by the diffusion method. Null → zero prior."
        ),
    )
    n_chains:    int   = Field(3,   ge=1, le=10, description="Diffusion chains to average.")
    cfg_guide_w: float = Field(3.0, ge=0, le=10, description="CFG guidance weight.")
    return_format: FORMAT_CHOICES = Field(
        "h5",
        description=(
            "**h5** - HDF5 file (`execution_plan` shape (2, THREADS, STRESSORS))  |  "
            "**json** - action tensor"
        ),
    )


class SeriesRequest(BaseModel):
    traces: list[dict[str, float]] = Field(
        ..., min_length=1,
        description=(
            "Ordered list of T resource-usage snapshots. One action is generated per "
            "snapshot. For the diffusion method the action at step t−1 is fed as "
            "prev_action at step t (autoregressive). Baselines are stateless."
        ),
    )
    method: METHOD_CHOICES = Field("diffusion")
    initial_prev_action: Optional[list[list[float]]] = Field(
        None,
        description=(
            f"Initial prev_action for t=0, shape ({ACTION_STRESSORS}, {ACTION_THREADS}), "
            "values in [0, 1]. Null → zeros."
        ),
    )
    generation_strategy: STRATEGY_CHOICES = Field(
        "autoregressive",
        description=(
            "**autoregressive** - each step uses the previous step's prediction as "
            "prev_action (default).  "
            "**parallel_refine** - pass 1: all steps predicted with prev_action=None; "
            "pass 2: all steps refined using the pass-1 predecessor as prev_action."
        ),
    )
    n_chains:      int   = Field(3,   ge=1, le=10)
    cfg_guide_w:   float = Field(3.0, ge=0, le=10)
    return_format: FORMAT_CHOICES = Field("h5")


class ProfileRequest(BaseModel):
    trace:       dict[str, float]
    method:      METHOD_CHOICES = Field("diffusion")
    prev_action: Optional[list[list[float]]] = None
    n_chains:    int   = Field(3, ge=1, le=10)
    cfg_guide_w: float = Field(3.0, ge=0, le=10)


class ProfileSeriesRequest(BaseModel):
    traces: list[dict[str, float]] = Field(
        ..., min_length=1,
        description=(
            "Ordered list of T resource-usage snapshots. T actions are generated "
            "and executed sequentially on the remote worker with "
            "MIMESYS_ITERS=1 MIMESYS_SLEEP=0. Returns T metric dicts."
        ),
    )
    method:              METHOD_CHOICES           = Field("diffusion")
    initial_prev_action: Optional[list[list[float]]] = None
    generation_strategy: STRATEGY_CHOICES = Field(
        "autoregressive",
        description=(
            "**autoregressive** - each step uses the previous prediction as prev_action.  "
            "**parallel_refine** - pass 1 with prev=None, pass 2 refined with pass-1 predecessor."
        ),
    )
    n_chains:            int   = Field(3, ge=1, le=10)
    cfg_guide_w:         float = Field(3.0, ge=0, le=10)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def _to_h5_bytes(actions: list[np.ndarray]) -> bytes:
    """
    Pack a list of action arrays into an HDF5 binary blob.

    ``execution_plan`` has shape (T+1, THREADS, STRESSORS) where slot 0 is
    all-zeros and slots 1..T are the generated actions.
    This matches the format expected by ``collect_mimesys_data.sh``.
    """
    T = len(actions)
    plans = np.zeros((T + 1, ACTION_THREADS, ACTION_STRESSORS), dtype=np.float32)
    for t, act in enumerate(actions):
        plans[t + 1] = act.T   # (STRESSORS, THREADS) → (THREADS, STRESSORS)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("execution_plan", data=plans)
    return buf.getvalue()


def _format_response(
    preds:       list[np.ndarray],
    fmt:         str,
    series_meta: Optional[dict] = None,
) -> Response:
    if fmt == "h5":
        return Response(
            content=_to_h5_bytes(preds),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=execution_plan.h5"},
        )
    if fmt == "json":
        body: dict = {}
        if series_meta:
            body.update(series_meta)
            body["actions"] = [p.tolist() for p in preds]
        else:
            body["action"] = preds[0].tolist()
        body["action_shape"]   = [ACTION_STRESSORS, ACTION_THREADS]
        body["stressor_names"] = STRESSOR_NAMES
        return JSONResponse(body)
    return Response(status_code=400, content="Invalid format")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    args = _state["args"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device

    print(f"[server] Checkpoint : {args.ckpt}")
    print(f"[server] Device     : {device}")

    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(_pkg_root, "conf"), version_base=None):
        cfg = compose("config", overrides=[f"+exps={args.exp}"])
    _state["cfg"] = cfg

    # Dataloader (normalization ranges + baseline training data)
    print("[server] Building dataloader...")
    dl = CustomDataLoader(
        cfg.data.data_path, cfg.data.test_data_path,
        max_time_steps=2, batch_size=1, test_batch_size=1, aug_factor=1,
    )

    # Build engine
    print("[server] Loading model and building baseline indices...")
    engine = InferenceEngine.from_checkpoint(
        ckpt_path=args.ckpt,
        cfg=cfg,
        dataloader=dl,
        device=device,
    )
    _state["engine"] = engine
    print(
        f"[server] Ready — trace_dim={len(METRIC_KEYS)}, "
        f"action=({ACTION_STRESSORS}x{ACTION_THREADS}), "
        f"epoch={engine.ckpt_meta['epoch']}"
    )

    _state["profiling_enabled"] = args.enable_profiling
    if args.enable_profiling:
        for h in cfg.data.test_machines:
            _worker_queue.put(h)
        print(f"[server] Remote profiling enabled — {_worker_queue.qsize()} workers in pool")

    yield
    print("[server] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Stress-Emulation Inference Service",
    description=(
        "Diffusion-model inference service: accepts hardware resource-usage traces "
        "and returns executable execution plans for workload emulation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Liveness check")
def health():
    engine: InferenceEngine = _state["engine"]
    return {
        "status":            "ok",
        "device":            _state["device"],
        "epoch":             engine.ckpt_meta.get("epoch"),
        "global_step":       engine.ckpt_meta.get("global_step"),
        "profiling_enabled": _state.get("profiling_enabled", False),
        "methods_available": list(METHOD_CHOICES.__args__),
    }


@app.get("/info", summary="Model, hardware, and configuration metadata")
def info():
    engine: InferenceEngine = _state["engine"]
    cfg = _state.get("cfg")
    return {
        "model": {
            "checkpoint":      engine.ckpt_meta.get("path"),
            "epoch":           engine.ckpt_meta.get("epoch"),
            "global_step":     engine.ckpt_meta.get("global_step"),
            "weight_source":   engine.ckpt_meta.get("weight_source"),
            "architecture":    "Temporal-UNet + Transformer ContextEncoder (DDPM)",
            "diffusion_steps": 25,
            "cfg_guidance":    True,
        },
        "input": {
            "trace_dim":         len(METRIC_KEYS),
            "trace_keys":        METRIC_KEYS,
            "prev_action_shape": [ACTION_STRESSORS, ACTION_THREADS],
            "normalization":     "linear min-max per metric (from training corpus)",
        },
        "output": {
            "action_shape":     [ACTION_STRESSORS, ACTION_THREADS],
            "stressor_names":   STRESSOR_NAMES,
            "h5_dataset_shape": ["T+1", ACTION_THREADS, ACTION_STRESSORS],
            "h5_dataset_name":  "execution_plan",
            "value_range":      "[0, 1] (thread-load fraction per stressor)",
        },
        "training": {
            "dataset":      "surrogate_v2 (CloudLab c220g2 profiling traces)",
            "augmentation": "5x intra-socket permutation + 10x high-IO oversampling",
            "hardware":     TRAINING_HARDWARE,
        },
        "methods": {
            "diffusion": {
                "description":      "Pretrained denoising diffusion model (best quality)",
                "autoregressive":   True,
                "training_required": True,
            },
            "nearest_neighbor": {
                "description":    "Return action of training sample with closest trace (L2)",
                "autoregressive": False,
            },
            "linear_interpolation": {
                "description":    "Inverse-distance-weighted blend of 5 nearest training actions",
                "autoregressive": False,
            },
            "single_stressor": {
                "description":    "Heuristic: map per-core CPU utilisation to the cpu stressor",
                "autoregressive": False,
            },
        },
        "profiling": {
            "enabled": _state.get("profiling_enabled", False),
            "workers": list(cfg.data.test_machines) if cfg else [],
        },
    }


@app.get("/metrics", summary="Supported input metrics with training-corpus ranges")
def metrics():
    """Lists all metric dimensions accepted in `trace` fields, with units and training ranges."""
    engine: InferenceEngine = _state["engine"]
    tr = engine.trace_range
    return {
        "metrics": [
            {
                "name":      k,
                "unit":      METRIC_UNITS.get(k, ""),
                "train_min": float(tr[k][0]) if k in tr else None,
                "train_max": float(tr[k][1]) if k in tr else None,
            }
            for k in METRIC_KEYS
        ],
        "total": len(METRIC_KEYS),
    }


@app.post(
    "/generate",
    summary="Generate a single-step execution plan from one resource-usage snapshot",
    responses={200: {"description": "HDF5 binary / JSON"}},
)
def generate(req: GenerateRequest):
    """
    Accepts one hardware resource-usage snapshot and returns an execution plan
    for the **next** measurement window.

    | `return_format` | Content-Type | Description |
    |---|---|---|
    | `h5` | `application/octet-stream` | HDF5 with `execution_plan` (2, THREADS, STRESSORS) |
    | `json` | `application/json` | Action tensor (STRESSORS x THREADS) in [0, 1] |
    """
    engine: InferenceEngine = _state["engine"]
    try:
        _, pred = engine.generate_step(
            req.trace, req.prev_action, req.method, req.n_chains, req.cfg_guide_w
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _format_response([pred], req.return_format)


@app.post(
    "/generate/series",
    summary="Generate a time-series of execution plans from a sequence of traces",
    responses={200: {"description": "HDF5 binary / JSON"}},
)
def generate_series(req: SeriesRequest):
    """
    Accepts T ordered resource-usage snapshots and returns T execution plans.

    **Diffusion method**: action at step *t* is fed as `prev_action` at step *t+1*
    (autoregressive temporal continuity).  Baselines are stateless.

    **HDF5 output**: `execution_plan` shape **(T+1, THREADS, STRESSORS)**, slot 0 = zero prior.

    **JSON output**: `{"T": T, "method": "...", "actions": [...]}`
    """
    engine: InferenceEngine = _state["engine"]
    try:
        strategy = getattr(req, "generation_strategy", "autoregressive")
        if strategy == "parallel_refine":
            preds = engine.generate_series_parallel_refine(
                req.traces, req.n_chains, req.cfg_guide_w,
            )
        else:
            preds = engine.generate_series(
                req.traces, req.method, req.initial_prev_action,
                req.n_chains, req.cfg_guide_w,
            )
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _format_response(preds, req.return_format,
                             series_meta={"T": len(preds), "method": req.method,
                                          "strategy": strategy})


@app.post(
    "/profile",
    summary="Submit a profiling job; returns a job_id immediately",
    status_code=202,
)
async def profile(req: ProfileRequest):
    """
    Submits a profiling job and returns a **job_id** immediately (HTTP 202).
    The job generates a plan, deploys it to a CloudLab worker, runs the
    benchmark, and collects measured hardware metrics — all in the background.

    **Requires** `--enable_profiling` at server startup.

    **Track progress** via SSE:
    ```
    GET /profile/jobs/{job_id}/stream
    ```
    Each event is a JSON object:
    ```json
    {"stage": "benchmarking", "pct": 55, "message": "[3/5] Processing batch 3/5..."}
    ```

    **Fetch final result** (blocks until done or returns current status):
    ```
    GET /profile/jobs/{job_id}/result
    ```

    Stages: `queued → generating_plan → packaging → connecting →
             transferring → benchmarking → collecting → parsing → done | error`
    """
    if not _state.get("profiling_enabled"):
        raise HTTPException(503, "Remote profiling disabled. Restart with --enable_profiling.")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "stage":   "queued",
        "pct":     0,
        "message": "Job queued",
        "result":  None,
        "error":   None,
        "events":  asyncio.Queue(),
    }
    loop = asyncio.get_event_loop()
    asyncio.create_task(_run_profile_job(job_id, req, loop))
    return {
        "job_id":     job_id,
        "stream_url": f"/profile/jobs/{job_id}/stream",
        "result_url": f"/profile/jobs/{job_id}/result",
    }


@app.get(
    "/profile/jobs/{job_id}/stream",
    summary="SSE stream of progress events for a profiling job",
)
async def profile_stream(job_id: str):
    """
    Server-Sent Events stream.  Each event is a JSON object:

    ```json
    {"stage": "benchmarking", "pct": 55, "message": "[3/5] Processing batch 3/5..."}
    ```

    The stream closes automatically when stage is `done` or `error`.
    """
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    async def _event_gen():
        q: asyncio.Queue = _jobs[job_id]["events"]
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["stage"] in ("done", "error"):
                break

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


@app.get(
    "/profile/jobs/{job_id}/result",
    summary="Poll the final result of a profiling job",
)
async def profile_result(job_id: str):
    """
    Returns the profiling result once the job completes.
    If the job is still running, returns current `{stage, pct, message}`.
    On error, returns HTTP 500 with the error message.
    """
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    job = _jobs[job_id]
    if job["error"]:
        raise HTTPException(500, job["error"])
    if job["result"] is not None:
        return job["result"]
    return {"status": "pending", "stage": job["stage"],
            "pct": job["pct"], "message": job["message"]}


@app.post(
    "/profile/series",
    summary="Submit a time-series profiling job; returns a job_id immediately",
    status_code=202,
)
async def profile_series(req: ProfileSeriesRequest):
    """
    Generates T actions from T traces (autoregressive diffusion), packs them into
    a single H5 execution plan, and runs the benchmark on a CloudLab worker with
    ``MIMESYS_ITERS=1 MIMESYS_SLEEP=0`` so each slot is executed
    exactly once without an inter-slot sleep.

    Returns a **job_id** immediately (HTTP 202). Track progress via SSE:
    ```
    GET /profile/jobs/{job_id}/stream
    ```

    The final result contains a list of T metric dicts — one per generated action.

    **Requires** ``--enable_profiling`` at server startup.
    """
    if not _state.get("profiling_enabled"):
        raise HTTPException(503, "Remote profiling disabled. Restart with --enable_profiling.")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "stage":   "queued",
        "pct":     0,
        "message": "Job queued",
        "result":  None,
        "error":   None,
        "events":  asyncio.Queue(),
    }
    loop = asyncio.get_event_loop()
    asyncio.create_task(_run_profile_series_job(job_id, req, loop))
    return {
        "job_id":     job_id,
        "stream_url": f"/profile/jobs/{job_id}/stream",
        "result_url": f"/profile/jobs/{job_id}/result",
    }


# ---------------------------------------------------------------------------
# Profile background job
# ---------------------------------------------------------------------------
async def _run_profile_job(job_id: str, req: ProfileRequest, loop: asyncio.AbstractEventLoop):
    """Async wrapper: runs the blocking profiling work in a thread pool."""
    job = _jobs[job_id]

    def emit(stage: str, message: str, pct: int):
        job["stage"]   = stage
        job["pct"]     = pct
        job["message"] = message
        event = {"stage": stage, "pct": pct, "message": message}
        asyncio.run_coroutine_threadsafe(job["events"].put(event), loop)

    try:
        result = await loop.run_in_executor(
            None, lambda: _profile_sync(req, emit)
        )
        job["result"] = result
        emit("done", "Profiling complete.", 100)
    except Exception as exc:
        job["error"] = str(exc)
        emit("error", str(exc), job["pct"])


def _profile_sync(req: ProfileRequest, emit) -> dict:
    """
    Blocking profiling pipeline.  Calls emit(stage, message, pct) at each
    milestone so the async layer can forward them as SSE events.
    """

    engine: InferenceEngine = _state["engine"]
    cfg = _state["cfg"]

    # ── 1. Generate plan ─────────────────────────────────────────────────────
    emit("generating_plan", "Running inference to generate execution plan...", 5)
    try:
        _, pred = engine.generate_step(
            req.trace, req.prev_action, req.method, req.n_chains, req.cfg_guide_w
        )
    except ValueError as e:
        raise RuntimeError(str(e))

    # ── 2. Package + acquire worker ──────────────────────────────────────────
    emit("packaging", "Packaging execution plan into archive...", 12)
    emit("queued", "Waiting for an available worker...", 13)
    hostname = _worker_queue.get(block=True)
    machine  = Machine.from_hostname(hostname)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan_000000.h5")
            prev_zeros = np.zeros((ACTION_THREADS, ACTION_STRESSORS), dtype=np.float32)
            with h5py.File(plan_path, "w") as f:
                f.create_dataset("execution_plan",
                                 data=np.stack([prev_zeros, pred.T], axis=0))
            zip_path = os.path.join(tmp, "chunk_0.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(plan_path, "plan_000000.h5")

            # ── 3. Connect ───────────────────────────────────────────────────
            emit("connecting", f"Connecting to {machine.hostname}...", 18)
            client, sftp_conn = machine.initialize_connection(
                username=cfg.profiler.user_name,
                private_key_path=cfg.profiler.private_key_path,
            )

            # ── 4. Transfer ──────────────────────────────────────────────────
            emit("transferring", f"Uploading execution plans to {machine.hostname}...", 25)
            remote_zip = (f"/users/{cfg.profiler.user_name}/fleetbench/"
                          f"execution_plans/chunk_0.zip")
            machine.file_transfer(scp=sftp_conn, file_path=zip_path, destination=remote_zip)
            machine.run_command(client=client, command=(
                f"cd /users/{cfg.profiler.user_name}/fleetbench/execution_plans/ "
                f"&& rm -f *.h5 && unzip -o chunk_0.zip && rm chunk_0.zip"
            ))

            # ── 5. Benchmark ─────────────────────────────────────────────────
            emit("benchmarking", "Starting benchmark on remote worker...", 30)
            _run_benchmark_streaming(
                client=client,
                command=(
                    f"cd /users/{cfg.profiler.user_name}/fleetbench && "
                    f"bash collect_mimesys_data.sh 2>&1"
                ),
                emit=emit,
            )

            # ── 6. Collect results ───────────────────────────────────────────
            emit("collecting", "Downloading result files from worker...", 90)
            result_dest = os.path.join(tmp, "results")
            os.makedirs(result_dest, exist_ok=True)
            _, stdout, _ = client.exec_command(
                f"ls /users/{cfg.profiler.user_name}/results/stats-plan_*.txt 2>/dev/null"
            )
            sftp = client.open_sftp()
            for rp in stdout.read().decode().strip().splitlines():
                if rp.strip():
                    sftp.get(rp.strip(),
                             os.path.join(result_dest, os.path.basename(rp.strip())))
            sftp.close()
            machine.close_connection(scp=sftp_conn, client=client)

            # ── 7. Parse ─────────────────────────────────────────────────────
            emit("parsing", "Parsing hardware metric traces...", 95)
            hw_metrics: dict = {}
            for fname in os.listdir(result_dest):
                _header, grouped_traces = parse_trace_file(os.path.join(result_dest, fname))
                if not grouped_traces:
                    continue
                res = process_trace_all(grouped_traces, include_aggregated_cpu=True)
                if res:
                    hw_metrics = {k: vals for k, vals in res.items()}
                    break
    finally:
        _worker_queue.put(hostname)

    return {
        "method":           req.method,
        "action":           pred.tolist(),
        "action_shape":     [ACTION_STRESSORS, ACTION_THREADS],
        "measured_metrics": hw_metrics,
        "worker":           machine.hostname,
    }


async def _run_profile_series_job(
    job_id: str, req: ProfileSeriesRequest, loop: asyncio.AbstractEventLoop
):
    """Async wrapper: runs the blocking time-series profiling work in a thread pool."""
    job = _jobs[job_id]

    def emit(stage: str, message: str, pct: int):
        job["stage"]   = stage
        job["pct"]     = pct
        job["message"] = message
        event = {"stage": stage, "pct": pct, "message": message}
        asyncio.run_coroutine_threadsafe(job["events"].put(event), loop)

    try:
        result = await loop.run_in_executor(
            None, lambda: _profile_series_sync(req, emit)
        )
        job["result"] = result
        emit("done", "Time-series profiling complete.", 100)
    except Exception as exc:
        job["error"] = str(exc)
        emit("error", str(exc), job["pct"])


def _profile_series_sync(req: ProfileSeriesRequest, emit) -> dict:
    """
    Blocking time-series profiling pipeline.

    Generates T actions from T traces (autoregressive), packs them into one H5
    with T+1 slots (slot 0 = zeros), runs the benchmark with
    MIMESYS_ITERS=1 MIMESYS_SLEEP=0, and parses the resulting
    T diff intervals as a list of T hardware-metric dicts.
    """

    engine: InferenceEngine = _state["engine"]
    cfg = _state["cfg"]
    T = len(req.traces)
    strategy = getattr(req, "generation_strategy", "autoregressive")

    # ── 1. Wait for a worker before doing any GPU work ───────────────────────
    # Acquiring the worker first prevents all batch jobs from piling up on the
    # GPU simultaneously — only as many plans are generated as there are free
    # remote workers, keeping machines busy.
    emit("queued", "Waiting for an available worker...", 5)
    hostname = _worker_queue.get(block=True)
    machine  = Machine.from_hostname(hostname)

    try:
        # ── 2. Generate T actions ────────────────────────────────────────────
        emit("generating_plan", f"Generating {T}-step plan (strategy={strategy})...", 10)
        try:
            if strategy == "parallel_refine":
                preds = engine.generate_series_parallel_refine(
                    req.traces, req.n_chains, req.cfg_guide_w,
                )
            else:
                preds = engine.generate_series(
                    req.traces, req.method, req.initial_prev_action,
                    req.n_chains, req.cfg_guide_w,
                )
        except ValueError as e:
            raise RuntimeError(str(e))

        # ── 3. Package into single H5 (T+1 slots) ───────────────────────────
        emit("packaging", f"Packaging {T+1}-slot execution plan...", 18)
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan_000000.h5")
            h5_data = np.zeros((T + 1, ACTION_THREADS, ACTION_STRESSORS), dtype=np.float32)
            for t, act in enumerate(preds):
                h5_data[t + 1] = act.T   # (STRESSORS, THREADS) → (THREADS, STRESSORS)
            with h5py.File(plan_path, "w") as f:
                f.create_dataset("execution_plan", data=h5_data)
            zip_path = os.path.join(tmp, "chunk_0.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(plan_path, "plan_000000.h5")

            # ── 4. Connect ───────────────────────────────────────────────────
            emit("connecting", f"Connecting to {machine.hostname}...", 22)
            client, sftp_conn = machine.initialize_connection(
                username=cfg.profiler.user_name,
                private_key_path=cfg.profiler.private_key_path,
            )

            # ── 5. Transfer ──────────────────────────────────────────────────
            emit("transferring", f"Uploading execution plans to {machine.hostname}...", 28)
            remote_zip = (f"/users/{cfg.profiler.user_name}/fleetbench/"
                          f"execution_plans/chunk_0.zip")
            machine.file_transfer(scp=sftp_conn, file_path=zip_path, destination=remote_zip)
            machine.run_command(client=client, command=(
                f"cd /users/{cfg.profiler.user_name}/fleetbench/execution_plans/ "
                f"&& rm -f *.h5 && unzip -o chunk_0.zip && rm chunk_0.zip"
            ))

            # ── 6. Benchmark — ITERS=1 SLEEP=0 ──────────────────────────────
            emit("benchmarking", "Starting time-series benchmark on remote worker...", 33)
            bench_lines = _run_benchmark_streaming(
                client=client,
                command=(
                    f"cd /users/{cfg.profiler.user_name}/fleetbench && "
                    f"MIMESYS_ITERS=1 MIMESYS_SLEEP=0 "
                    f"bash collect_mimesys_data.sh 2>&1"
                ),
                emit=emit,
            )

            # ── 7. Collect results ───────────────────────────────────────────
            emit("collecting", "Downloading result files from worker...", 90)
            result_dest = os.path.join(tmp, "results")
            os.makedirs(result_dest, exist_ok=True)
            _, stdout, _ = client.exec_command(
                f"ls /users/{cfg.profiler.user_name}/results/stats-plan_*.txt 2>/dev/null"
            )
            sftp = client.open_sftp()
            for rp in stdout.read().decode().strip().splitlines():
                if rp.strip():
                    sftp.get(rp.strip(),
                             os.path.join(result_dest, os.path.basename(rp.strip())))
            log_content = "\n".join(bench_lines) + "\n"
            try:
                with sftp.open(f"/users/{cfg.profiler.user_name}/benchmark.log", "w") as rf:
                    rf.write(log_content)
            except Exception as e:
                print(f"[series] WARNING: failed to write benchmark.log: {e}", flush=True)
            sftp.close()
            machine.close_connection(scp=sftp_conn, client=client)

            # ── 8. Parse — each diff is one time-step ───────────────────────
            emit("parsing", "Parsing time-series hardware metric traces...", 95)
            series_metrics: list[dict] = []
            for fname in sorted(os.listdir(result_dest)):
                _header, grouped_traces = parse_trace_file(os.path.join(result_dest, fname))
                if not grouped_traces:
                    continue
                res = process_trace_all(grouped_traces, include_aggregated_cpu=True)
                if not res:
                    continue
                n_steps = len(next(iter(res.values())))
                for step in range(1, n_steps):
                    d = {k: float(vals[step]) for k, vals in res.items()}
                    series_metrics.append(d)
                break
    finally:
        _worker_queue.put(hostname)

    return {
        "method":         req.method,
        "strategy":       strategy,
        "T":              T,
        "actions":        [p.tolist() for p in preds],
        "action_shape":   [ACTION_STRESSORS, ACTION_THREADS],
        "series_metrics": series_metrics,
        "worker":         machine.hostname,
    }


class BatchProfileRequest(BaseModel):
    jobs:                list[ProfileSeriesRequest]
    method:              METHOD_CHOICES  = Field("diffusion")
    generation_strategy: STRATEGY_CHOICES = Field("autoregressive")
    n_chains:            int   = Field(3, ge=1, le=10)
    cfg_guide_w:         float = Field(3.0, ge=0, le=10)


@app.post(
    "/profile/batch",
    summary="Submit a batch of time-series profiling jobs; returns immediately",
    status_code=202,
)
async def profile_batch(req: BatchProfileRequest):
    """
    Returns a batch_id immediately (HTTP 202).  Plan generation and SSH dispatch
    run in the background.  Poll GET /profile/batch/{batch_id}/status for progress.

    Stages: generating_plans → dispatching → (per-job: queued/connecting/benchmarking/…/done)
    """
    if not _state.get("profiling_enabled"):
        raise HTTPException(503, "Remote profiling disabled. Restart with --enable_profiling.")

    N = len(req.jobs)
    batch_id = uuid.uuid4().hex[:8]
    _batches[batch_id] = {
        "job_ids":  [],
        "total":    N,
        "stage":    "generating_plans",
        "pct":      0,
        "message":  f"Generating plans for {N} jobs...",
    }

    loop = asyncio.get_event_loop()
    asyncio.create_task(_run_batch(batch_id, req, loop))

    return {
        "batch_id":   batch_id,
        "total":      N,
        "status_url": f"/profile/batch/{batch_id}/status",
    }


async def _run_batch(batch_id: str, req: BatchProfileRequest, loop: asyncio.AbstractEventLoop):
    """Background task: generate all plans (batched), then dispatch SSH jobs."""
    engine: InferenceEngine = _state["engine"]
    batch = _batches[batch_id]

    # Resolve per-job settings
    merged_jobs: list[ProfileSeriesRequest] = []
    for job_req in req.jobs:
        merged_jobs.append(ProfileSeriesRequest(
            traces              = job_req.traces,
            method              = job_req.method if job_req.method != "diffusion" else req.method,
            generation_strategy = (job_req.generation_strategy
                                   if job_req.generation_strategy != "autoregressive"
                                   else req.generation_strategy),
            n_chains            = job_req.n_chains    if job_req.n_chains    != 3   else req.n_chains,
            cfg_guide_w         = job_req.cfg_guide_w if job_req.cfg_guide_w != 3.0 else req.cfg_guide_w,
            initial_prev_action = job_req.initial_prev_action,
        ))

    strategy    = merged_jobs[0].generation_strategy if merged_jobs else "autoregressive"
    method      = merged_jobs[0].method if merged_jobs else req.method
    n_chains    = merged_jobs[0].n_chains
    cfg_guide_w = merged_jobs[0].cfg_guide_w
    max_T       = max(len(j.traces) for j in merged_jobs) if merged_jobs else 0

    def on_progress(step: int, total: int):
        batch["pct"]     = int(step / total * 100)
        batch["message"] = f"Generating plans: step {step}/{total}"

    try:
        if strategy == "parallel_refine":
            def on_progress(pass_num: int, total_passes: int):
                batch["pct"]     = int(pass_num / total_passes * 100)
                batch["message"] = f"Generating plans: pass {pass_num}/{total_passes}"

            all_preds: list = await loop.run_in_executor(
                None,
                lambda: engine.generate_series_batch_parallel_refine(
                    [j.traces for j in merged_jobs],
                    n_chains=n_chains,
                    cfg_guide_w=cfg_guide_w,
                    progress_cb=on_progress,
                ),
            )
        else:
            all_preds = await loop.run_in_executor(
                None,
                lambda: engine.generate_series_batch(
                    [j.traces for j in merged_jobs],
                    method=method,
                    initial_prev_actions=[j.initial_prev_action for j in merged_jobs],
                    n_chains=n_chains,
                    cfg_guide_w=cfg_guide_w,
                    progress_cb=on_progress,
                ),
            )
    except Exception as exc:
        batch["stage"]   = "error"
        batch["message"] = str(exc)
        print(f"[batch {batch_id}] plan generation failed: {exc}", flush=True)
        return

    batch["stage"]   = "dispatching"
    batch["pct"]     = 100
    batch["message"] = f"Plans ready — dispatching {len(merged_jobs)} jobs to workers"
    print(f"[batch {batch_id}] plans generated, dispatching {len(merged_jobs)} jobs", flush=True)

    job_ids: list[str] = []
    for merged, preds in zip(merged_jobs, all_preds):
        job_id = uuid.uuid4().hex[:8]
        _jobs[job_id] = {
            "stage":    "queued",
            "pct":      0,
            "message":  "Plan ready, waiting for worker",
            "result":   None,
            "error":    None,
            "events":   asyncio.Queue(),
            "batch_id": batch_id,
        }
        asyncio.create_task(_run_profile_series_job_with_plan(job_id, merged, preds, loop))
        job_ids.append(job_id)

    batch["job_ids"] = job_ids
    batch["stage"]   = "running"
    batch["message"] = f"All {len(job_ids)} jobs dispatched"


@app.get("/profile/batch/{batch_id}/status", summary="Poll status of a batch job")
def batch_status(batch_id: str):
    if batch_id not in _batches:
        raise HTTPException(404, f"Batch {batch_id!r} not found")
    batch = _batches[batch_id]
    batch_stage = batch.get("stage", "unknown")

    counts = {"done": 0, "error": 0, "queued": 0, "running": 0}
    statuses: dict = {}
    for job_id in batch["job_ids"]:
        job = _jobs.get(job_id, {})
        stage = job.get("stage", "unknown")
        statuses[job_id] = {
            "stage":   stage,
            "pct":     job.get("pct", 0),
            "message": job.get("message", ""),
            "worker":  (job.get("result") or {}).get("worker"),
        }
        if stage == "done":
            counts["done"] += 1
        elif stage == "error":
            counts["error"] += 1
        elif stage == "queued":
            counts["queued"] += 1
        else:
            counts["running"] += 1

    dispatched = len(batch["job_ids"])
    all_done = (batch_stage not in ("generating_plans", "dispatching", "error")
                and dispatched == batch["total"]
                and counts["done"] + counts["error"] == dispatched)
    return {
        "batch_id":     batch_id,
        "total":        batch["total"],
        "batch_stage":  batch_stage,
        "batch_pct":    batch.get("pct", 0),
        "batch_message": batch.get("message", ""),
        "workers_free": _worker_queue.qsize(),
        **counts,
        "all_done":     all_done,
        "job_statuses": statuses,
    }


async def _run_profile_series_job_with_plan(
    job_id: str,
    req:    ProfileSeriesRequest,
    preds:  list,
    loop:   asyncio.AbstractEventLoop,
):
    """Async wrapper for SSH/benchmark work when plans are already generated."""
    job = _jobs[job_id]

    def emit(stage: str, message: str, pct: int):
        job["stage"]   = stage
        job["pct"]     = pct
        job["message"] = message
        asyncio.run_coroutine_threadsafe(
            job["events"].put({"stage": stage, "pct": pct, "message": message}), loop
        )

    try:
        result = await loop.run_in_executor(
            None, lambda: _profile_series_sync_with_plan(req, preds, emit)
        )
        job["result"] = result
        emit("done", "Time-series profiling complete.", 100)
    except Exception as exc:
        job["error"] = str(exc)
        emit("error", str(exc), job["pct"])


def _profile_series_sync_with_plan(
    req:   ProfileSeriesRequest,
    preds: list,
    emit,
) -> dict:
    """SSH/benchmark pipeline with a pre-generated plan — no GPU inference."""
    cfg = _state["cfg"]
    T   = len(req.traces)
    strategy = getattr(req, "generation_strategy", "autoregressive")

    emit("queued", "Waiting for an available worker...", 5)
    hostname = _worker_queue.get(block=True)
    machine  = Machine.from_hostname(hostname)

    try:
        emit("packaging", f"Packaging {T+1}-slot execution plan...", 15)
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan_000000.h5")
            h5_data = np.zeros((T + 1, ACTION_THREADS, ACTION_STRESSORS), dtype=np.float32)
            for t, act in enumerate(preds):
                h5_data[t + 1] = act.T
            with h5py.File(plan_path, "w") as f:
                f.create_dataset("execution_plan", data=h5_data)
            zip_path = os.path.join(tmp, "chunk_0.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(plan_path, "plan_000000.h5")

            emit("connecting", f"Connecting to {machine.hostname}...", 22)
            client, sftp_conn = machine.initialize_connection(
                username=cfg.profiler.user_name,
                private_key_path=cfg.profiler.private_key_path,
            )

            emit("transferring", f"Uploading to {machine.hostname}...", 28)
            remote_zip = (f"/users/{cfg.profiler.user_name}/fleetbench/"
                          f"execution_plans/chunk_0.zip")
            machine.file_transfer(scp=sftp_conn, file_path=zip_path, destination=remote_zip)
            machine.run_command(client=client, command=(
                f"cd /users/{cfg.profiler.user_name}/fleetbench/execution_plans/ "
                f"&& rm -f *.h5 && unzip -o chunk_0.zip && rm chunk_0.zip"
            ))

            emit("benchmarking", "Starting benchmark on remote worker...", 33)
            bench_lines = _run_benchmark_streaming(
                client=client,
                command=(
                    f"cd /users/{cfg.profiler.user_name}/fleetbench && "
                    f"MIMESYS_ITERS=1 MIMESYS_SLEEP=0 "
                    f"bash collect_mimesys_data.sh 2>&1"
                ),
                emit=emit,
            )

            emit("collecting", "Downloading result files...", 90)
            result_dest = os.path.join(tmp, "results")
            os.makedirs(result_dest, exist_ok=True)
            _, stdout, _ = client.exec_command(
                f"ls /users/{cfg.profiler.user_name}/results/stats-plan_*.txt 2>/dev/null"
            )
            sftp = client.open_sftp()
            for rp in stdout.read().decode().strip().splitlines():
                if rp.strip():
                    sftp.get(rp.strip(),
                             os.path.join(result_dest, os.path.basename(rp.strip())))
            sftp.close()
            machine.close_connection(scp=sftp_conn, client=client)

            emit("parsing", "Parsing time-series hardware metric traces...", 95)
            series_metrics: list[dict] = []
            for fname in sorted(os.listdir(result_dest)):
                _header, grouped_traces = parse_trace_file(os.path.join(result_dest, fname))
                if not grouped_traces:
                    continue
                res = process_trace_all(grouped_traces, include_aggregated_cpu=True)
                if not res:
                    continue
                n_steps = len(next(iter(res.values())))
                for step in range(1, n_steps):
                    series_metrics.append({k: float(vals[step]) for k, vals in res.items()})
                break
    finally:
        _worker_queue.put(hostname)

    return {
        "method":         req.method,
        "strategy":       strategy,
        "T":              T,
        "actions":        [p.tolist() for p in preds],
        "action_shape":   [ACTION_STRESSORS, ACTION_THREADS],
        "series_metrics": series_metrics,
        "worker":         machine.hostname,
    }


def _run_benchmark_streaming(client, command: str, emit) -> list:
    """
    Execute *command* over SSH, reading stdout line by line and forwarding
    benchmark-script progress markers as emit() calls.

    The remote ``collect_mimesys_data.sh`` already prints:
        "Processing batch X/N..."
        "Running commands on batch X..."
        "Batch processing complete."
    These are parsed to derive a [30, 88] % progress range.

    Returns all collected output lines so the caller can persist them as a
    remote log file via SFTP.
    """
    _, stdout, stderr = client.exec_command(command)

    collected_lines = []

    # Paramiko's stdout channel supports readline() for streaming.
    for raw_line in iter(stdout.readline, ""):
        line = raw_line.rstrip()
        collected_lines.append(line)
        if not line:
            continue

        # "Processing batch 3/10..."
        if line.startswith("Processing batch"):
            try:
                token = line.split()[2]          # "3/10..."
                cur, total = token.rstrip(".").split("/")
                cur, total = int(cur), int(total)
                pct = 30 + int((cur - 1) / max(total, 1) * 58)   # 30→88
                emit("benchmarking", f"[{cur}/{total}] {line}", pct)
            except (ValueError, IndexError):
                emit("benchmarking", line, 50)

        elif line.startswith("Running commands"):
            emit("benchmarking", line, 55)

        elif "Batch processing complete" in line:
            emit("benchmarking", line, 88)

        else:
            emit("benchmarking", line, 50)

    return collected_lines