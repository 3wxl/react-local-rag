"""本地 ONNX Qwen2.5-0.5B-Instruct 生成（与浏览器 Lite 模式同一个模型文件）。

为什么自己写解码循环：Transformers.js 导出的 ONNX 图带完整的 KV cache 输入
（`past_key_values.{i}.key/value` + 输出 `present.{i}.*`），但 Python 没有现成的
"读这个目录就能跑"的封装。循环本身很短：贪心取 argmax、把 present 喂回 past。

与浏览器侧的差异（写进 evidence_snapshot，避免"看起来一样其实不一样"）：
- 采样：这里固定**贪心**（do_sample=False），可复现；浏览器问答默认带温度采样
- 运行时：onnxruntime CPU EP vs onnxruntime-web WASM，实测向量余弦 0.99+（数值差异）
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import Completion, ModelUnavailableError

#: Qwen2.5 的 EOS（<|im_end|>）与填充符（<|endoftext|>）
EOS_IDS = (151645, 151643)


class QwenOnnxGenerator:
    """本地 ONNX 生成器（惰性加载 + KV cache 增量解码）。"""

    name = "onnx"
    model = "Qwen2.5-0.5B-Instruct(onnx/q4)"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._session = None
        self._tokenizer = None

    # ---------------- 可用性 ----------------

    @property
    def _model_path(self) -> Path:
        return self.model_dir / "onnx" / "model_q4.onnx"

    @property
    def _tokenizer_path(self) -> Path:
        return self.model_dir / "tokenizer.json"

    def available(self) -> bool:
        return self._model_path.exists() and self._tokenizer_path.exists()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if not self.available():
            raise ModelUnavailableError(
                f"本地生成模型缺失：{self._model_path}（下载见 README「模型权重」一节）"
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path))
        self._session = ort.InferenceSession(
            str(self._model_path), providers=["CPUExecutionProvider"]
        )

    # ---------------- 生成 ----------------

    def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
        self._ensure_loaded()
        started = time.time()
        ids = self._tokenizer.encode(prompt).ids
        prompt_len = len(ids)

        # KV cache 初值：形状 (batch=1, kv_heads=2, past_len=0, head_dim=64)
        past = self._empty_past()
        generated: list[int] = []
        next_ids = np.asarray([ids], dtype=np.int64)
        past_len = 0
        logits = None

        for _ in range(max_new_tokens):
            inputs = {
                "input_ids": next_ids,
                "attention_mask": np.ones((1, past_len + next_ids.shape[1]), dtype=np.int64),
                "position_ids": np.arange(
                    past_len, past_len + next_ids.shape[1], dtype=np.int64
                )[None, :],
            }
            inputs.update(past)
            outputs = self._session.run(None, inputs)
            logits = outputs[0]
            # present.{i}.key/value 的顺序与输入 past 一致，直接复用
            present = outputs[1:]
            past = {
                name: value
                for name, value in zip(
                    (n for n in self._past_names()), present
                )
            }
            past_len += next_ids.shape[1]
            token = int(np.argmax(logits[0, -1]))
            if token in EOS_IDS:
                break
            generated.append(token)
            next_ids = np.asarray([[token]], dtype=np.int64)

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return Completion(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=len(generated),
            model=self.model,
            provider=self.name,
            latency_ms=int((time.time() - started) * 1000),
            sampling={
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "eos_ids": list(EOS_IDS),
                "logits_shape": list(logits.shape) if logits is not None else None,
            },
        )

    # ---------------- 内部 ----------------

    def _past_names(self) -> list[str]:
        """past_key_values 输入名，按层序（与 present 输出同序）。"""
        names: list[str] = []
        for i in range(64):  # 层数从图里读，64 是安全上界
            key = f"past_key_values.{i}.key"
            if any(inp.name == key for inp in self._session.get_inputs()):
                names.append(key)
                names.append(f"past_key_values.{i}.value")
            else:
                break
        return names

    def _empty_past(self) -> dict[str, np.ndarray]:
        """空 KV cache：形状取自图定义 (batch, kv_heads, past_len, head_dim)。"""
        shape_by_name = {i.name: i.shape for i in self._session.get_inputs()}
        out: dict[str, np.ndarray] = {}
        for name in self._past_names():
            shape = [1 if isinstance(d, str) else d for d in shape_by_name[name]]
            # past_sequence_length 维置 0
            out[name] = np.zeros((1, shape[1], 0, shape[3]), dtype=np.float32)
        return out
