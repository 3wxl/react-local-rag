"""分块与索引版本参数（与 apps/web/src/utils/chunk.ts 逐字段对齐）。

这些常量是**算法的一部分**，不是可调配置：改任何一个都会让 contentHash / 向量空间变化。
Python 侧不允许"顺手调优"，要改必须先改 TS 并重跑 tuner（见 chunk.ts 里的扫描记录）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .fnv import fnv1a32


@dataclass(frozen=True)
class ChunkOptions:
    """DEFAULT_CHUNK_OPTIONS 的 Python 对应物（字段顺序参与索引版本计算，勿动）。"""

    max_chunk_tokens: int = 256
    min_chunk_chars: int = 100
    max_chunk_chars: int = 600
    sentence_overlap: int = 2


DEFAULT_CHUNK_OPTIONS = ChunkOptions()

#: BGE 查询指令（bge-small-zh-v1.5 官方中文 instruction）：只有查询加，文档与校验句不加
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

#: embedding 前缀方案描述（参与索引版本计算）
EMBEDDING_PREFIX_SCHEME = f"passage:none|query:{BGE_QUERY_INSTRUCTION}|verify:none"

#: embedding 模型名（参与索引版本计算：换模型旧向量不可复用）
EMBEDDING_MODEL_NAME = "Xenova/bge-small-zh-v1.5"


def _js_stringify_compact(value: object) -> str:
    """等价于 JS `JSON.stringify(value)`（无空格分隔符、非 ASCII 不转义）。

    `json.dumps` 默认带空格（`[1, "a"]`）且默认转义非 ASCII（`"\\u4e2d"`），
    两者都会算出不同的哈希 —— 索引版本跨端就对不上了。
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def compute_index_version(options: ChunkOptions = DEFAULT_CHUNK_OPTIONS) -> str:
    """索引版本 = hash(分块参数 + 前缀方案 + 模型名)。与 TS `computeIndexVersion` 同值。"""
    canonical = _js_stringify_compact(
        [
            options.max_chunk_tokens,
            options.min_chunk_chars,
            options.max_chunk_chars,
            options.sentence_overlap,
            EMBEDDING_PREFIX_SCHEME,
            EMBEDDING_MODEL_NAME,
        ]
    )
    return fnv1a32(canonical)


#: 当前索引版本（与 TS 的 CURRENT_INDEX_VERSION 必须相等；不等即索引口径漂移）
CURRENT_INDEX_VERSION = compute_index_version()
