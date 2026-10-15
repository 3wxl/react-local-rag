"""检索后处理（apps/web/src/utils/retrieval.ts 的移植）。

流程顺序**不可换**（每一步都依赖上一步的产物）：

1. RRF 融合：两路各自排名后用 Σ w/(k+rank) 相加（不做分数归一化，规避量纲不一致）
2. 邻近合并：同文档 ord 连续的命中合成一个窗口（overlap 片段不再各占一个坑）
3. 窗口聚合：窗口内的最高分为该窗口的分，向量取 RRF 最高那条（供 MMR）
4. contentHash 去重：相同内容只留最高分的一条
5. 门限 + 自适应 K：相关度门限过滤（调用方据此走「文档未提及」拒答）
6. MMR 保多样：λ·relevance − (1−λ)·与已选集合的最大相似度
7. 上下文扩展：对最终 ≤ topK 条向两侧扩邻块（小块检索 / 大块喂 LLM）

排序稳定性与并列处理都用"首个最大值"规则（对应 JS 稳定排序 + 严格 `>` 比较）。
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field, replace
from typing import Sequence

from .fnv import fnv1a32
from .jsstr import js_len

#: RRF 融合常数 k（RRF 论文推荐值，tune 扫描确认维持默认）
RRF_K = 60

#: 分数完全相同的判定容差（精排输出无区分度检测）
RERANK_DEGENERATE_EPSILON = 1e-6


@dataclass
class RetrievalDoc:
    """参与检索的文档块（TextChunk + 可选向量；vector=None 表示向量路降级）。"""

    chunk_id: str
    doc_id: str
    content: str
    doc_name: str
    ord: int
    heading_path: list[str | None]
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    content_hash: str
    vector: Sequence[float] | None = None
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""


@dataclass
class RetrievalOptions:
    """检索参数（默认值 = TS DEFAULT_RETRIEVAL_OPTIONS，逐个都有实测依据，勿随手改）。"""

    top_k: int = 5
    #: 余弦相关度门限：向量可用时只看它（2026-10-02 实测的零误拒最宽点）
    min_vec_score: float = 0.5
    #: BM25 豁免排名：**仅向量降级时生效**（向量可用时豁免会让负样本靠通用词挤进前三）
    bm25_exempt_rank: int = 3
    #: IDF 覆盖率门限：默认关闭（0）。口语化查询与负样本在该信号上无法区分，勿设非零
    min_idf_coverage: float = 0.0
    #: 相对门限：RRF 分低于最高分 × ratio 的命中被丢弃
    relative_cut_ratio: float = 0.4
    #: 邻近合并窗口的字符上限
    max_window_chars: int = 1200
    #: 上下文扩展半径（命中块前后各取多少块）
    window_radius: int = 1
    #: MMR 多样性权重（1 = 只看相关度 = 关闭 MMR）
    mmr_lambda: float = 0.7
    #: MMR 候选池大小
    mmr_pool_size: int = 20
    rrf_k: int = RRF_K
    vec_weight: float = 1.0
    bm25_weight: float = 1.0


DEFAULT_RETRIEVAL_OPTIONS = RetrievalOptions()


@dataclass
class FusedHit:
    """融合后的原始命中（合并/去重前）。"""

    index: int
    doc: RetrievalDoc
    vec_score: float | None
    vec_rank: int | None
    bm25_score: float
    bm25_rank: int
    rrf_score: float


@dataclass
class SearchHit:
    """最终返回给调用方的检索命中（小块检索 / 大块喂 LLM）。"""

    chunk_ids: list[str]
    window_chunk_ids: list[str]
    doc_id: str
    doc_name: str
    ord_start: int
    ord_end: int
    heading_path: list[str | None]
    page_start: int
    page_end: int
    content: str
    matched_content: str
    rrf_score: float
    vec_score: float | None
    bm25_score: float
    vec_rank: int | None
    bm25_rank: int
    relevance: float = 0.0
    #: MMR 用的向量（不进入对外序列化）
    vector: Sequence[float] | None = None


@dataclass
class ThresholdResult:
    hits: list[SearchHit]
    no_relevant_hit: bool
    dropped_count: int


@dataclass
class RetrievalResult(ThresholdResult):
    fused: list[FusedHit] = field(default_factory=list)


# ---------------- 1. RRF 融合 ----------------


def scores_to_ranks(scores: Sequence[float], include_zero: bool) -> list[int]:
    """原始分 → 排名（1 起，0 = 零分未上榜；同分并列）。"""
    n = len(scores)
    if n == 0:
        return []
    ranks = [0] * n
    # 稳定排序：同分保持原下标顺序（JS Array.sort 自 ES2019 起稳定）
    order = sorted(
        (i for i in range(n) if include_zero or scores[i] > 0),
        key=lambda i: -scores[i],
    )
    for r, idx in enumerate(order):
        if r > 0 and scores[order[r]] == scores[order[r - 1]]:
            ranks[idx] = ranks[order[r - 1]]
        else:
            ranks[idx] = r + 1
    return ranks


def to_float32(values: Sequence[float]) -> array:
    """按 float32 存一份（对应 TS 的 Float32Array.from）。

    这一步不是"精度优化"，而是**语义对齐**：固化向量在 JSON 里是高精度小数，
    TS 侧先转 float32 再做点积（float32 元素、float64 累加），
    Python 若直接用 float64 相乘，相关系数会差 ~1e-9 —— 门限判定在边界上就可能翻转。
    """
    return array("f", values)


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """点积（归一化向量的余弦相似度等于点积）。

    逐元素顺序累加：与 TS 的 `for (i...) sum += a[i]*b[i]` 同序，
    numpy 的 pairwise 求和会改变末位精度 —— 门限判定在边界上就靠这点精度。
    """
    total = 0.0
    for x, y in zip(a, b):
        total += float(x) * float(y)
    return total


def rrf_fuse(
    docs: Sequence[RetrievalDoc],
    bm25_scores: Sequence[float],
    vec_scores: Sequence[float] | None,
    *,
    k: int = RRF_K,
    vec_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[FusedHit]:
    """RRF 融合：Σ w/(k+rank)。两路排名口径不同：向量路含零分，BM25 路不含。"""
    bm25_ranks = scores_to_ranks(bm25_scores, False)
    vec_ranks = scores_to_ranks(vec_scores, True) if vec_scores is not None else None

    fused: list[FusedHit] = []
    for i, doc in enumerate(docs):
        vr = vec_ranks[i] if vec_ranks is not None else None
        rrf_score = 0.0
        if vr and vr > 0:
            rrf_score += vec_weight / (k + vr)
        if bm25_ranks[i] > 0:
            rrf_score += bm25_weight / (k + bm25_ranks[i])
        fused.append(
            FusedHit(
                index=i,
                doc=doc,
                vec_score=vec_scores[i] if vec_scores is not None else None,
                vec_rank=vr if vr and vr > 0 else None,
                bm25_score=bm25_scores[i] if i < len(bm25_scores) else 0.0,
                bm25_rank=bm25_ranks[i] if i < len(bm25_ranks) else 0,
                rrf_score=rrf_score,
            )
        )
    return fused


# ---------------- 2. 邻近合并 ----------------


def overlap_length(a: str, b: str) -> int:
    """a 的末尾与 b 的开头的最长重叠长度（分块自带整句重叠，拼接时去重）。"""
    for length in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:length]):
            return length
    return 0


def append_without_overlap(a: str, b: str) -> str:
    return a + b[overlap_length(a, b) :]


def merge_neighbors(
    hits: Sequence[FusedHit], max_window_chars: int = 1200
) -> list[list[FusedHit]]:
    """同一 docId 且 ord 连续的命中合并成一个窗口（按 docId 首次出现顺序输出）。"""
    by_doc: dict[str, list[FusedHit]] = {}
    for hit in hits:
        by_doc.setdefault(hit.doc.doc_id, []).append(hit)

    windows: list[list[FusedHit]] = []
    for group in by_doc.values():
        ordered = sorted(group, key=lambda h: h.doc.ord)  # 稳定排序
        current: list[FusedHit] = []
        current_chars = 0
        for hit in ordered:
            if not current:
                current = [hit]
                current_chars = js_len(hit.doc.content)
                continue
            prev = current[-1]
            contiguous = hit.doc.ord == prev.doc.ord + 1
            fits = current_chars + js_len(hit.doc.content) <= max_window_chars
            if contiguous and fits:
                current.append(hit)
                current_chars += js_len(hit.doc.content)
            else:
                windows.append(current)
                current = [hit]
                current_chars = js_len(hit.doc.content)
        if current:
            windows.append(current)
    return windows


def _join_chunks(contents: Sequence[str]) -> str:
    if not contents:
        return ""
    out = contents[0]
    for content in contents[1:]:
        out = append_without_overlap(out, content)
    return out


def window_to_hit(window: Sequence[FusedHit]) -> SearchHit:
    """窗口内全部信息合成一条 SearchHit（核心=精确命中；窗口扩张见 expand_hit）。"""
    ordered = sorted(window, key=lambda h: h.doc.ord)
    best = window[0]
    for hit in window:
        if hit.rrf_score > best.rrf_score:
            best = hit
    matched_content = _join_chunks([h.doc.content for h in ordered])
    vec_score: float | None = None
    for hit in window:
        if hit.vec_score is None:
            continue
        vec_score = hit.vec_score if vec_score is None else max(vec_score, hit.vec_score)
    first = ordered[0].doc
    last = ordered[-1].doc
    chunk_ids = [h.doc.chunk_id for h in ordered]
    bm25_rank = 0
    for hit in window:
        if hit.bm25_rank > 0:
            bm25_rank = hit.bm25_rank if bm25_rank == 0 else min(bm25_rank, hit.bm25_rank)
    return SearchHit(
        chunk_ids=chunk_ids,
        # 未扩展前，窗口 = 命中块自身（expand_hit 负责向两侧扩邻块）
        window_chunk_ids=list(chunk_ids),
        doc_id=first.doc_id,
        doc_name=first.doc_name,
        ord_start=first.ord,
        ord_end=last.ord,
        heading_path=first.heading_path,
        page_start=first.page_start,
        page_end=last.page_end,
        content=matched_content,
        matched_content=matched_content,
        rrf_score=best.rrf_score,
        vec_score=vec_score,
        bm25_score=max((h.bm25_score for h in window), default=0.0),
        vec_rank=best.vec_rank,
        bm25_rank=bm25_rank,
        relevance=0.0,  # 由 apply_threshold 统一计算
        vector=best.doc.vector,
    )


def expand_hit(
    hit: SearchHit,
    doc_chunks: Sequence[RetrievalDoc],
    radius: int = 1,
    max_chars: int = 1200,
) -> SearchHit:
    """上下文扩展（小块检索 / 大块喂 LLM）：向两侧各扩 radius 块，超限向命中区收拢。"""
    if radius <= 0 or not doc_chunks:
        return hit
    matched = set(hit.chunk_ids)
    matched_idx = [i for i, d in enumerate(doc_chunks) if d.chunk_id in matched]
    if not matched_idx:
        return hit

    core_lo = matched_idx[0]
    core_hi = matched_idx[-1]
    lo = max(0, core_lo - radius)
    hi = min(len(doc_chunks) - 1, core_hi + radius)

    def size(low: int, high: int) -> int:
        return sum(js_len(d.content) for d in doc_chunks[low : high + 1])

    while lo < core_lo and size(lo, hi) > max_chars:
        lo += 1
    while hi > core_hi and size(lo, hi) > max_chars:
        hi -= 1

    window = doc_chunks[lo : hi + 1]
    return replace(
        hit,
        window_chunk_ids=[d.chunk_id for d in window],
        content=_join_chunks([d.content for d in window]),
        ord_start=window[0].ord,
        ord_end=window[-1].ord,
        page_start=window[0].page_start,
        page_end=window[-1].page_end,
    )


# ---------------- 3. 去重 ----------------


def hash_of(text: str) -> str:
    """文本哈希（FNV-1a 32 位，与 chunk.ts 的 contentHash 同算法）。"""
    return fnv1a32(text)


def dedupe_by_hash(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """contentHash 去重：完全相同内容只保留 RRF 分最高的一条（保持首次出现顺序）。"""
    seen: dict[str, SearchHit] = {}
    for hit in hits:
        key = hash_of(hit.content)
        prev = seen.get(key)
        if prev is None:
            seen[key] = hit
        elif hit.rrf_score > prev.rrf_score:
            seen[key] = hit
    return list(seen.values())


# ---------------- 4. MMR 保多样 ----------------


def _similarity(a: SearchHit, b: SearchHit) -> float:
    """两跳命中的语义相似度（任一条无向量时按 0 处理，退化为纯相关度排序）。"""
    # 注意用 `is None` 判断：向量可能是 numpy 数组，"真值判断"会抛 ambiguous truth value
    if a.vector is None or b.vector is None:
        return 0.0
    if len(a.vector) == 0 or len(b.vector) == 0:
        return 0.0
    return dot_product(a.vector, b.vector)


def mmr_select(
    hits: list[SearchHit], limit: int, lam: float = 0.7
) -> list[SearchHit]:
    """每轮挑「λ·相关度 − (1−λ)·与已选集合的最大相似度」最高的一条。"""
    if lam >= 1 or len(hits) <= 1:
        return hits[:limit]
    pool = list(hits)
    selected: list[SearchHit] = []
    while pool and len(selected) < limit:
        best_idx = 0
        best_score = -math.inf
        for i, cand in enumerate(pool):
            max_sim = 0.0
            for s in selected:
                max_sim = max(max_sim, _similarity(cand, s))
            score = lam * cand.relevance - (1 - lam) * max_sim
            if score > best_score:  # 严格大于 → 并列取先出现者（与 TS 一致）
                best_score = score
                best_idx = i
        selected.append(pool[best_idx])
        del pool[best_idx]
    return selected


# ---------------- 5. 门限 + 自适应 K ----------------


def _compute_relevance(hit: SearchHit, vec_available: bool) -> float:
    """归一化相关度：向量可用取余弦（+BM25 命中加成），降级按 BM25 排名折算经验分。"""
    if vec_available and hit.vec_score is not None:
        keyword_bonus = 0.05 if hit.bm25_rank > 0 else 0.0
        return min(1.0, hit.vec_score + keyword_bonus)
    if hit.bm25_rank > 0:
        return max(0.0, 0.6 - (hit.bm25_rank - 1) * 0.05)
    return 0.0


def apply_threshold(
    hits: list[SearchHit],
    options: RetrievalOptions,
    *,
    vec_available: bool,
    idf_coverage: float | None = None,
    top_k: int | None = None,
) -> ThresholdResult:
    """门限 + 自适应 K + 检索侧拒答信号（精排不参与拒答判定）。"""
    min_vec = options.min_vec_score
    ratio = options.relative_cut_ratio
    exempt_rank = options.bm25_exempt_rank
    min_idf = options.min_idf_coverage
    limit = options.top_k if top_k is None else top_k

    # 信号 1：查询词在语料里几乎找不到 → 直接拒答（唯一不依赖候选质量的信号）
    if (idf_coverage if idf_coverage is not None else 1.0) < min_idf:
        return ThresholdResult(hits=[], no_relevant_hit=True, dropped_count=len(hits))

    for hit in hits:
        hit.relevance = _compute_relevance(hit, vec_available)

    # 相关度门限：向量可用只看余弦（不给 BM25 豁免）；降级纯 BM25 时用豁免排名
    if vec_available:
        passed = [h for h in hits if (h.vec_score or 0.0) >= min_vec]
    else:
        passed = [h for h in hits if 0 < h.bm25_rank <= exempt_rank]

    if not passed:
        return ThresholdResult(hits=[], no_relevant_hit=True, dropped_count=len(hits))

    top_rrf = max(h.rrf_score for h in passed)
    cut = top_rrf * ratio
    kept = [h for h in passed if h.rrf_score >= cut]
    capped = kept[:limit]
    return ThresholdResult(
        hits=capped, no_relevant_hit=False, dropped_count=len(hits) - len(capped)
    )


# ---------------- 主流程 ----------------


def post_process_retrieval(
    docs: Sequence[RetrievalDoc],
    bm25_scores: Sequence[float],
    vec_scores: Sequence[float] | None,
    options: RetrievalOptions = DEFAULT_RETRIEVAL_OPTIONS,
    *,
    idf_coverage: float | None = None,
) -> RetrievalResult:
    """检索后处理全流程（顺序与 TS 一致，见模块 docstring）。"""
    fused = rrf_fuse(
        docs,
        bm25_scores,
        vec_scores,
        k=options.rrf_k,
        vec_weight=options.vec_weight,
        bm25_weight=options.bm25_weight,
    )
    candidates = [h for h in fused if h.rrf_score > 0]
    ordered = sorted(candidates, key=lambda h: -h.rrf_score)

    windows = merge_neighbors(ordered, options.max_window_chars)
    hits = [window_to_hit(w) for w in windows]
    hits.sort(key=lambda h: -h.rrf_score)

    unique = dedupe_by_hash(hits)
    pool = unique[: options.mmr_pool_size]

    gated = apply_threshold(
        pool,
        options,
        vec_available=vec_scores is not None,
        idf_coverage=idf_coverage,
        top_k=options.mmr_pool_size,
    )
    if not gated.hits:
        return RetrievalResult(
            hits=[],
            no_relevant_hit=True,
            dropped_count=len(hits),
            fused=fused,
        )

    selected = mmr_select(gated.hits, options.top_k, options.mmr_lambda)

    by_doc: dict[str, list[RetrievalDoc]] = {}
    for d in docs:
        by_doc.setdefault(d.doc_id, []).append(d)
    for group in by_doc.values():
        group.sort(key=lambda d: d.ord)

    expanded = [
        expand_hit(
            h,
            by_doc.get(h.doc_id, []),
            options.window_radius,
            options.max_window_chars,
        )
        for h in selected
    ]
    return RetrievalResult(
        hits=expanded,
        no_relevant_hit=False,
        dropped_count=len(hits) - len(selected),
        fused=fused,
    )
