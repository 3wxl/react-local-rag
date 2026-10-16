"""领域对象 → 契约 JSON（snake_case，字段名与 docs/api-contract.md 一一对应）。

集中放一处的原因：引用卡片与逐句校验在 `/v1/retrieve`、`/v1/answer`、`/v1/answer/{id}/trace`
里都要序列化，散在各路由里迟早会出现"同一个字段两处写法不同"。
"""

from __future__ import annotations

from ..parity.citation import ContextRef
from ..parity.verify import VerificationResult


def context_to_dict(c: ContextRef) -> dict:
    """引用卡片 → 契约 §0.1 Citation。"""
    return {
        "index": c.index,
        "doc_id": c.doc_id,
        "doc_name": c.doc_name,
        "heading_path": c.heading_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "score": c.score,
        "relevance": c.relevance,
        "chunk_ids": c.chunk_ids,
        "window_chunk_ids": c.window_chunk_ids,
        "matched_content": c.matched_content,
        "content": c.content,
    }


def verification_to_dict(v: VerificationResult) -> dict:
    """逐句校验 → 契约 §2 verification。"""
    return {
        "source": v.source,
        "scoped_chunk_count": v.scoped_chunk_count,
        # 引用闭环收敛情况（M2）：未标注 / 越界 / 是否全部角标都能回跳到编号段落
        "convergence": {
            "uncited_claims": v.convergence.uncited_claims,
            "out_of_range_claims": v.convergence.out_of_range_claims,
            "citations_subset_of_passages": v.convergence.citations_subset_of_passages,
        },
        "sentences": [
            {
                "text": s.text,
                "level": s.level,
                "score": s.score,
                "semantic": s.semantic,
                "lexical": s.lexical,
                "unknown_numbers": s.unknown_numbers,
                "citations": s.citations,
                "invalid_citations": s.invalid_citations,
                "uncited": s.uncited,
                "trivial": s.trivial,
            }
            for s in v.sentences
        ],
    }
