"""Precompile the complete NDLOCR PARSEQ MIGraphX cache profile.

Run this with the same environment used by the service.  It intentionally
writes the profile only after all three recognizers and all batch buckets have
completed successfully.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from migraphx_cache import make_profile, write_profile
from parseq import PARSEQ

MODELS = [
    ("parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx", (256, 16)),
    ("parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx", (384, 16)),
    ("parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx", (768, 16)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="amdgpu", choices=["amdgpu", "cuda"])
    ap.add_argument("--max-batch", type=int, default=16)
    args = ap.parse_args()
    if args.max_batch < 1 or args.max_batch & (args.max_batch - 1):
        raise SystemExit("--max-batch must be a positive power of two")

    with (ROOT / "src/config/NDLmoji.yaml").open(encoding="utf-8") as f:
        chars = list(yaml.safe_load(f)["model"]["charset_test"])
    paths = [ROOT / "src/model" / name for name, _ in MODELS]
    buckets = []
    b = 1
    while b <= args.max_batch:
        buckets.append(b)
        b *= 2

    for name, size in MODELS:
        print(f"[compile] {name} buckets={buckets}", flush=True)
        recognizer = PARSEQ(str(ROOT / "src/model" / name), chars,
                            original_size=size, device=args.device,
                            use_fp16=args.device != "CPU",
                            max_batch=args.max_batch)
        # PARSEQ.__init__ performs the bucket warmup.  Keep one explicit
        # inference as a completion check before releasing the session.
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        result = recognizer.read_batch([image] * args.max_batch)
        if len(result) != args.max_batch:
            raise RuntimeError(f"completion check failed for {name}")
        del recognizer
        gc.collect()
        print(f"[compile] complete {name}", flush=True)

    profile = make_profile(paths, device=args.device, use_fp16=True,
                           max_batch=args.max_batch, buckets=buckets)
    path = write_profile(profile)
    print(f"[compile] ALL COMPLETE profile={path}", flush=True)


if __name__ == "__main__":
    main()
