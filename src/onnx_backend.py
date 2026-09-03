"""ONNX Runtime backend selection shared by detector and recognizer."""

import onnxruntime


def available_providers() -> list[str]:
    return list(onnxruntime.get_available_providers())


def provider_for(device: str) -> str | None:
    name = device.casefold()
    if name in ("cpu", ""):
        return None
    if name in ("cuda", "nvidia"):
        return "CUDAExecutionProvider"
    if name in ("amdgpu", "amd", "rocm", "migraphx"):
        return "MIGraphXExecutionProvider"
    if name in ("vulkan", "vk"):
        return "VulkanExecutionProvider"
    raise ValueError(f"unknown ONNX Runtime device: {device}")


def session_providers(device: str) -> list:
    provider = provider_for(device)
    if provider is None:
        return ["CPUExecutionProvider"]
    providers = available_providers()
    if provider not in providers:
        raise RuntimeError(f"{provider} is unavailable; installed providers: {providers}")
    options = {"arena_extend_strategy": "kSameAsRequested"} if provider == "CUDAExecutionProvider" else {}
    return [(provider, options), "CPUExecutionProvider"]


def is_accelerated(device: str) -> bool:
    return provider_for(device) is not None
