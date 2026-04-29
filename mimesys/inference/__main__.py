"""
Entry point: python -m mimesys.inference

Examples
--------
  # Basic
  python -m mimesys.inference \\
      --ckpt /path/to/diffusion-epoch=999.ckpt

  # Custom port + experiment config
  python -m mimesys.inference \\
      --ckpt /path/to/last.ckpt \\
      --exp prev_study_film \\
      --port 8000

  # With remote profiling enabled
  python -m mimesys.inference \\
      --ckpt /path/to/last.ckpt \\
      --enable_profiling

  # Force CPU
  python -m mimesys.inference \\
      --ckpt /path/to/last.ckpt \\
      --device cpu
"""

import argparse
import torch
import uvicorn

from mimesys.inference.server import app, _state


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m mimesys.inference",
        description="Stress-emulation inference server",
    )
    p.add_argument("--ckpt",             required=True,
                   help="Path to model checkpoint (.ckpt)")
    p.add_argument("--port",             type=int, default=8000,
                   help="HTTP port (default: 8000)")
    p.add_argument("--host",             default="0.0.0.0",
                   help="Bind address (default: 0.0.0.0)")
    p.add_argument("--exp",              default="stress_ng",
                   help="Hydra experiment config name (default: stress_ng)")
    p.add_argument("--enable_profiling", action="store_true",
                   help="Enable POST /profile — requires CloudLab SSH access")
    p.add_argument("--device",           default=None,
                   help="Force device: cuda or cpu (default: cuda if available)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _state["args"] = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
