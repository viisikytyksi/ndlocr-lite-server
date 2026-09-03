"""Benchmark batch vs sequential across ONNX Runtime backends."""
import argparse
import sys, os, time
REPO = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import yaml
import onnxruntime as ort
from parseq import PARSEQ

with open(str(REPO / 'src/config/NDLmoji.yaml'), encoding='utf-8') as f:
    charlist = list(yaml.safe_load(f)['model']['charset_test'])

model_path = str(REPO / 'src/model/parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx')

np.random.seed(0)
N = 16
imgs = [np.random.randint(0, 255, (16, 256, 3), dtype=np.uint8) for _ in range(N)]

def bench(device, n_warmup=2, n_repeat=5, use_fp16=False, max_batch=0):
    p = PARSEQ(model_path, charlist, original_size=(256, 16), device=device, use_fp16=use_fp16, max_batch=max_batch)
    label = f"{device}" + (" fp16" if use_fp16 else " fp32")
    print(f"\n[{label}] dynamic={p._has_dynamic_batch}, model={__import__('pathlib').Path(p._preferred_path()).name}")

    # warmup
    for _ in range(n_warmup):
        p.read_batch(imgs)

    # sequential (read one by one)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        for img in imgs:
            p.read(img)
    seq_t = (time.perf_counter() - t0) / n_repeat
    print(f"  Sequential ({N} imgs): {seq_t*1000:.1f} ms  ({seq_t/N*1000:.1f} ms/img)")

    # batch
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        p.read_batch(imgs)
    bat_t = (time.perf_counter() - t0) / n_repeat
    print(f"  Batch     ({N} imgs): {bat_t*1000:.1f} ms  ({bat_t/N*1000:.1f} ms/img)")
    print(f"  Speedup: {seq_t/bat_t:.2f}x")

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["CPU", "cuda", "amdgpu", "vulkan", "all"], default="CPU")
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--max-batch", type=int, default=0)
parser.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
args = parser.parse_args()

devices = ["CPU", "amdgpu", "vulkan", "cuda"] if args.device == "all" else [args.device]
provider_map = {"cuda": "CUDAExecutionProvider", "amdgpu": "MIGraphXExecutionProvider", "vulkan": "VulkanExecutionProvider"}
providers = ort.get_available_providers()
for device in devices:
    provider = provider_map.get(device)
    if provider and provider not in providers:
        print(f"\n{device}: unavailable ({provider})")
        continue
    bench(device, n_warmup=args.warmup, n_repeat=args.repeat,
          use_fp16=(args.precision == "fp16" or (args.precision == "auto" and device != "CPU")), max_batch=args.max_batch)
