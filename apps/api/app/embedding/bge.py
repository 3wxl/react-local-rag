"""本地 ONNX 向量化（bge-small-zh-v1.5，与浏览器端同一份权重）。

对齐三件事（错一个，向量就与固化文件不可比）：
1. **权重文件**：`apps/web/public/models/Xenova/bge-small-zh-v1.5/onnx/model_quantized.onnx`
   —— 浏览器 WASM 后端默认 dtype=q8 就是它，Node 固化脚本也用 q8
2. **前缀**：只有查询加中文 instruction，文档（passage）与校验句（verify）不加
   （见 parity/chunk_params.py 的 EMBEDDING_PREFIX_SCHEME）
3. **池化**：mean pooling（按 attention mask 加权）+ L2 归一化 → 余弦退化为点积

模型是**惰性加载**的：没有权重文件的环境（CI）也能跑全部检索类测试与 Harness，
只是 `/v1/answer` 的生成分支会返回 503 model_unavailable。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..parity.chunk_params import BGE_QUERY_INSTRUCTION, EMBEDDING_MODEL_NAME

#: 单批文本数（与浏览器 worker 的 EMBED_BATCH_SIZE 无关：那边是 WASM，这边是本地 CPU）
DEFAULT_BATCH = 16

#: bge-small-zh 的位置上限（超过会被截断；正常块 ≤256 token，不会触发）
MAX_LENGTH = 512


def apply_embed_prefix(text: str, kind: str) -> str:
    """向量化前缀（与 apps/web/src/utils/embedPrefix.ts 同规则）。"""
    if kind == "query":
        return BGE_QUERY_INSTRUCTION + text
    return text


class BgeOnnxEmbedder:
    """bge-small-zh ONNX 编码器（线程安全：ONNX Runtime 会话内部串行）。"""

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = DEFAULT_BATCH,
        enabled: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        #: 显式关闭向量路（无模型环境 / 只想建库跑 BM25 的场景），不是"猜"出来的状态
        self.enabled = enabled
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    # ---------------- 模型加载 ----------------

    @property
    def model_id(self) -> str:
        return EMBEDDING_MODEL_NAME

    @property
    def available(self) -> bool:
        return (
            self.enabled
            and self._model_path().exists()
            and self._tokenizer_path().exists()
        )

    def _model_path(self) -> Path:
        return self.model_dir / "onnx" / "model_quantized.onnx"

    def _tokenizer_path(self) -> Path:
        return self.model_dir / "tokenizer.json"

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort  # 惰性导入：没装/没权重也能 import 本模块
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path()))
            self._tokenizer.enable_truncation(max_length=MAX_LENGTH)
            self._session = ort.InferenceSession(
                str(self._model_path()), providers=["CPUExecutionProvider"]
            )

    # ---------------- 编码 ----------------

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """返回 (N, dim) 的 float32 归一化向量。"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure_loaded()
        prefixed = [apply_embed_prefix(t, kind) for t in texts]
        out: list[np.ndarray] = []
        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i : i + self.batch_size]
            encodings = [self._tokenizer.encode(t) for t in batch]
            max_len = max(len(e.ids) for e in encodings)
            input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            attention = np.zeros((len(batch), max_len), dtype=np.int64)
            token_types = np.zeros((len(batch), max_len), dtype=np.int64)
            for row, enc in enumerate(encodings):
                ids = enc.ids
                input_ids[row, : len(ids)] = ids
                attention[row, : len(ids)] = 1
            hidden = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention,
                    "token_type_ids": token_types,
                },
            )[0]
            # mean pooling（按 attention mask 加权）→ L2 归一化，与 transformers.js 一致
            mask = attention[:, :, None].astype(np.float32)
            summed = (hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / counts
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append((pooled / np.clip(norms, 1e-12, None)).astype(np.float32))
        return np.vstack(out)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "passage")

    def embed_verify(self, texts: list[str]) -> np.ndarray:
        """校验句与文档同处一个向量空间：不加前缀（旧实现加 query 前缀是错的）。"""
        return self.embed(texts, "verify")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], "query")[0]


_embedder: BgeOnnxEmbedder | None = None


def get_embedder(model_dir: Path) -> BgeOnnxEmbedder:
    """进程内单例（与浏览器端的常驻 worker 模型同理：加载一次、复用）。"""
    global _embedder
    if _embedder is None or _embedder.model_dir != Path(model_dir):
        _embedder = BgeOnnxEmbedder(Path(model_dir))
    return _embedder
