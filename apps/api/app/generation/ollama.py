"""Ollama 生成通道（HTTP /api/chat）。

适用场景：想换模型/换规模（7B/14B）而不改代码，或本机已有 Ollama 常驻。
与其它通道共用同一个 Prompt（`parity/answer_prompt.py`），所以切通道不会改变引用口径。

代价（要写进对比报告）：模型不再是 Lite 模式那一个 → 双端差异里混入了"模型不同"，
不再是纯工程对比。默认仍走本地 ONNX，Ollama 需要显式指定 `LLM_PROVIDER=ollama`。
"""

from __future__ import annotations

import time

import httpx

from .base import Completion, ModelUnavailableError


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        *,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout

    def available(self) -> bool:
        """探活 + 确认模型已拉取（未拉取时给出可执行的修复提示）。"""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
        except Exception:  # noqa: BLE001 —— 探活失败就是不可用
            return False
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        return any(n == self.model or n.startswith(f"{self.model}:") for n in names)

    def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
        if not self.available():
            raise ModelUnavailableError(
                f"Ollama 不可用或未拉取模型 {self.model}："
                f"先启动 ollama serve，再执行 ollama pull {self.model}"
            )
        started = time.time()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": True,  # Prompt 已经是 Qwen chat 模板（与 Lite 模式同一份）
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": max_new_tokens,
            },
        }
        resp = httpx.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return Completion(
            text=data.get("response", ""),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            model=self.model,
            provider=self.name,
            latency_ms=int((time.time() - started) * 1000),
            sampling={
                "temperature": self.temperature,
                "seed": self.seed,
                "max_new_tokens": max_new_tokens,
            },
        )
