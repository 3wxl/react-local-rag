"""向量化通道：本地 ONNX（与浏览器 Lite 模式同一份权重）+ 固化向量（CI / 评测用）。"""

from .bge import BgeOnnxEmbedder, get_embedder
from .fixed import FixedEmbeddings, load_fixed_embeddings

__all__ = ["BgeOnnxEmbedder", "FixedEmbeddings", "get_embedder", "load_fixed_embeddings"]
