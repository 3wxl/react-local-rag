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
    """生成通道协议。

    `stream` 是**可选能力**：支持的通道实现它，不支持的就不实现（调用方用
    `getattr(provider, "stream", None)` 探测）。为什么用可选而不是协议强制：
    ONNX 通道的解码循环逐 token 贪心、无法给出干净的增量分片，硬塞一个"一次性
    返回全文"的假流只会让客户端以为拿到了真的流。契约里 token 事件因此标 best-effort。
    """

    name: str
    model: str

    def available(self) -> bool: ...

    def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion: ...

    # 可选成员（不写进 Protocol，避免强制实现）：
    #   def stream(self, prompt: str, *, max_new_tokens: int = 512,
    #              on_delta: Callable[[str], None] | None = None) -> Completion
    # 语义：逐片回调 on_delta（原始文本，调用方负责 strip_think_tags），
    # 返回值与 complete() 同构（含真实用量），保证"流式只是多了一条输出通道"。


class ModelUnavailableError(RuntimeError):
    """模型不可用（权重缺失 / 服务未启动 / 运行时不可用）→ 契约里的 503 model_unavailable。"""
