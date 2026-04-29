"""
mimesys.inference
====================
FastAPI inference service for the stress-emulation diffusion model.

Quick start
-----------
  python -m mimesys.inference \\
      --ckpt /path/to/diffusion-epoch=999.ckpt \\
      --port 8000

  curl http://localhost:8000/health
  curl http://localhost:8000/info | python3 -m json.tool
  curl -X POST http://localhost:8000/generate \\
       -H "Content-Type: application/json" \\
       -d '{"trace":{"io":3000,"l3_cache_usage_socket_0":9000},"return_format":"json"}'
"""

from mimesys.inference.baselines import NearestNeighbor, LinearInterpolation, SingleStressor
from mimesys.inference.engine import InferenceEngine

__all__ = ["InferenceEngine", "NearestNeighbor", "LinearInterpolation", "SingleStressor"]
