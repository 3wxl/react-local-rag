"""生成通道：本地 ONNX / Ollama / 假模型（测试用），共用同一份 Prompt。"""

from .base import Completion, LlmProvider, ModelUnavailableError
from .factory import build_provider
from .fake import FakeProvider
from .ollama import OllamaProvider
from .onnx_qwen import QwenOnnxGenerator

__all__ = [
    "Completion",
    "FakeProvider",
    "LlmProvider",
    "ModelUnavailableError",
    "OllamaProvider",
    "QwenOnnxGenerator",
    "build_provider",
]
