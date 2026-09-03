"""Compare NDLOCR-Lite PARSEQ inference across ONNX Runtime backends."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_backend import available_providers, provider_for
from parseq import PARSEQ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["cpu", "cuda", "vulkan", "amdgpu", "all"], default="all")
    parser.add_argument("--model", type=Path, default=ROOT / "src/model/parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    names = ["cpu", "cuda", "vulkan", "amdgpu"] if args.provider == "all" else [args.provider]
    for name in names:
        if name != "cpu" and provider_for(name) not in available_providers():
            print(f"{name}: SKIP ({provider_for(name)} unavailable)")
            continue
        recognizer = PARSEQ(str(args.model), [""] * 2048, device=name, max_batch=1, use_fp16=True)
        images = [
            np.zeros((32, 384, 3), dtype=np.uint8)
            for _ in range(args.batch)
        ]
        recognizer.read_batch(images)  # warmup
        started = time.perf_counter()
        for _ in range(args.repeat):
            recognizer.read_batch(images)
        elapsed = time.perf_counter() - started
        print(f"{name}: {args.batch * args.repeat / elapsed:.2f} lines/s ({elapsed / args.repeat * 1000:.1f} ms/batch)")


if __name__ == "__main__":
    main()
