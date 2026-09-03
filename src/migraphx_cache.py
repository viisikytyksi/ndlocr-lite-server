"""MIGraphX cache profile generation and validation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROFILE_NAME = "cache-profile.json"
RELEVANT_ENV = (
    "ORT_MIGRAPHX_CACHE_PATH",
    "ORT_MIGRAPHX_MODEL_CACHE_PATH",
    "MIOPEN_FIND_MODE",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "MIOPEN_USER_DB_PATH",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_profile(models: list[Path], *, device: str, use_fp16: bool,
                 max_batch: int, buckets: list[int]) -> dict[str, Any]:
    import onnxruntime as ort

    return {
        "schema": 1,
        "device": device,
        "precision": "fp16" if use_fp16 else "fp32",
        "max_batch": max_batch,
        "buckets": buckets,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "models": [
            {"path": str(p.resolve()), "sha256": _sha256(p)}
            for p in models
        ],
        "environment": {name: os.environ.get(name, "") for name in RELEVANT_ENV},
    }


def profile_path() -> Path | None:
    value = os.environ.get("ORT_MIGRAPHX_MODEL_CACHE_PATH", "")
    return Path(value).expanduser() / PROFILE_NAME if value else None


def write_profile(profile: dict[str, Any]) -> Path:
    path = profile_path()
    if path is None:
        raise RuntimeError("ORT_MIGRAPHX_MODEL_CACHE_PATH is not set")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def validate_profile(expected: dict[str, Any]) -> tuple[bool, str]:
    path = profile_path()
    if path is None:
        return True, "cache profile disabled (ORT_MIGRAPHX_MODEL_CACHE_PATH unset)"
    if not path.exists():
        return False, f"cache profile missing: {path}"
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"cache profile unreadable: {path}: {exc}"
    if actual != expected:
        return False, f"cache profile mismatch: {path}"
    return True, f"cache profile OK: {path}"
