"""生成通道选择（配置驱动，默认本地 ONNX）。"""

from __future__ import annotations

from pathlib import Path

from .base import LlmProvider, ModelUnavailableError
from .fake import FakeProvider
from .ollama import OllamaProvider
from .onnx_qwen import QwenOnnxGenerator


def build_provider(
    provider: str,
    *,
    onnx_dir: Path,
    ollama_model: str,
    ollama_base_url: str,
) -> LlmProvider:
    """按配置构造通道；未知值按本地 ONNX 处理（默认口径）。"""
    name = (provider or "onnx").lower()
    if name == "fake":
        return FakeProvider()
    if name == "ollama":
        return OllamaProvider(ollama_model, ollama_base_url)
    if name in ("onnx", "local", "qwen"):
        return QwenOnnxGenerator(onnx_dir)
    raise ModelUnavailableError(f"未知的 LLM_PROVIDER：{provider}")
