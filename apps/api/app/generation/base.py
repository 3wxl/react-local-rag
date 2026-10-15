"""生成通道的统一接口。

服务端要能跑三种通道，且**都走同一个 Prompt**（parity/answer_prompt.py）：
- `onnx`：本地 ONNX Qwen2.5-0.5B-Instruct（与浏览器 Lite 模式同一个模型 → 双端对比口径最干净）
- `ollama`：本地 Ollama 服务（HTTP），换模型/换规模时用
- `fake`：确定性假模型（测试与 CI 用，不改口径也不下载权重）

「模型不可用」必须是一等状态（返回 503 model_unavailable），
而不是启动失败 —— 检索、Harness、跨语言一致性都不依赖模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    """一次生成的结果（含可归因的成本信息，喂给 Harness 的预算门槛）。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    #: 采样参数快照（进 evidence_snapshot：同一条 trace 必须能解释"当时怎么生成的"）
    sampling: dict = field(default_factory=dict)


class LlmProvider(Protocol):
    """生成通道协议。"""

    name: str
    model: str

    def available(self) -> bool: ...

    def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion: ...


class ModelUnavailableError(RuntimeError):
    """模型不可用（权重缺失 / 服务未启动 / 运行时不可用）→ 契约里的 503 model_unavailable。"""
