"""检索后处理（apps/web/src/utils/retrieval.ts 的移植）。

流程顺序**不可换**（每一步都依赖上一步的产物）：

1. RRF 融合：两路各自排名后用 Σ w/(k+rank) 相加（不做分数归一化，规避量纲不一致）
2. 邻近合并：同文档 ord 连续的命中合成一个窗口（overlap 片段不再各占一个坑）
3. 窗口聚合：窗口内的最高分为该窗口的分，向量取 RRF 最高那条（供 MMR）
4. contentHash 去重：相同内容只留最高分的一条
5. 门限 + 自适应 K：多信号融合拒绝（IDF 覆盖率 ∪ 最高余弦 ∪ 余弦对比度 ∪ BM25 最高分，
   **并集、且仅在向量可用时启用**）+ 相关度门限过滤（调用方据此走「文档未提及」拒答）
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
    #: IDF 覆盖率门限（提示位，**不参与检索侧粗拒**，理由同 min_bm25）
    min_idf_coverage: float = 0.2277
    #: 查询级「最高余弦」门限（检索侧粗拒的语义路）。
    #: 2026-10-15 取 0.53：受保护人群（q+m）实测最低余弦 0.5427，故对 q+m 仍零误拒（余量 0.013），
    #: 同时把两条口语化难例（s006 0.535 / s011 0.541）放进来 —— 旧值 0.5426 正好把它们拒掉。
    min_max_cosine: float = 0.53
    #: 查询级「余弦对比度」门限（最高余弦 − 全语料余弦均值，零误拒切点 0.077628）
    min_cos_gap: float = 0.0776
    #: 查询级「BM25 最高分」——**已不参与检索侧粗拒**（2026-10-15 起降为提示信号）。
    #: 它与 IDF 覆盖率都无法区分口语化提问与否样本（两组分布重叠），放进并集只会把 s 组整体拒掉。
    #: 值保留在档：响应里的 refusal_signals 要如实报告取值，答案侧也要这条参考线。
    min_bm25: float = 8.2246
    #: 相对门限：RRF 分低于最高分 × ratio 的命中被丢弃
    relative_cut_ratio: float = 0.4
    #: 邻近合并窗口的字符上限
    max_window_chars: int = 1200
    #: 上下文扩展半径（命中块前后各取多少块）
    window_radius: int = 1
    #: MMR 多样性权重。**1.0 = 关闭 MMR**（2026-10-15 起）。
    #: 引入查询扩展后实测反转：λ=1.0 + 扩展开时 q 0.92 / s 0.50 / m 1.00，
    #: 而 λ=0.9 + 扩展开只有 s 0.14 —— MMR 的结果侧多样性会抵消多查询融合的收益
    #: （把正确答案当"重复"压下去）。多样性改由查询扩展提供（问法多样而非结果多样）。
    mmr_lambda: float = 1.0
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


@dataclass
class RefusalSignals:
    """查询级拒答信号快照（由全量两路原始分算出，与 TS computeRefusalSignals 同口径）。

    max_cosine / cos_gap 参与粗拒；max_bm25 / idf_coverage 是**提示信号**（不参与），
    但必须回传：答案侧与排障要用，且"当时看到的取值"不该事后无法复核。
    """

    max_cosine: float | None = None
    cos_gap: float | None = None
    max_bm25: float | None = None
    idf_coverage: float | None = None


@dataclass
class QueryVariantScores:
    """一个查询变体的两路原始分（查询扩展的多路输入，与 TS QueryVariantScores 同构）。"""

    bm25_scores: Sequence[float]
    vec_scores: Sequence[float] | None
    idf_coverage: float | None = None
    #: 该变体在融合里的相对权重（拓展变体 < 1）
    weight: float = 1.0


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


def rrf_fuse_runs(
    docs: Sequence[RetrievalDoc],
    runs: Sequence[QueryVariantScores],
    *,
    k: int = RRF_K,
    vec_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[FusedHit]:
    """多路 RRF 融合（查询扩展用）。`rrf_fuse` 是"只有一路"的特例。

    **只有第一路（原查询）决定命中上的分数与排名字段**，后续路只累加 RRF 分 ——
    vec_score/bm25_rank 既是展示值，也是接纳判据与 MMR 的相关度输入；
    让改写路抬高它们等于让"猜的问法"替用户的问题接纳候选（实测：q 组 0.88 → 0.84）。
    改写只做一件事：把"文档里有没有这件事"的信号抬高（见 best_refusal_signals）。
    """
    prepared = [
        (
            scores_to_ranks(run.bm25_scores, False),
            scores_to_ranks(run.vec_scores, True) if run.vec_scores is not None else None,
            run,
        )
        for run in runs
    ]
    primary_ranks, primary_vec_ranks, primary = prepared[0]

    fused: list[FusedHit] = []
    for i, doc in enumerate(docs):
        rrf_score = 0.0
        for bm25_ranks, vec_ranks, run in prepared:
            vr = vec_ranks[i] if vec_ranks is not None else None
            if vr and vr > 0:
                rrf_score += (vec_weight * run.weight) / (k + vr)
            if bm25_ranks[i] > 0:
                rrf_score += (bm25_weight * run.weight) / (k + bm25_ranks[i])
        vr = primary_vec_ranks[i] if primary_vec_ranks is not None else None
        fused.append(
            FusedHit(
                index=i,
                doc=doc,
                vec_score=primary.vec_scores[i] if primary.vec_scores is not None else None,
                vec_rank=vr if vr and vr > 0 else None,
                bm25_score=primary.bm25_scores[i] if i < len(primary.bm25_scores) else 0.0,
                bm25_rank=primary_ranks[i] if i < len(primary_ranks) else 0,
                rrf_score=rrf_score,
            )
        )
    return fused


def best_refusal_signals(variants: Sequence[QueryVariantScores]) -> RefusalSignals:
    """逐信号取各变体的最优值（与 TS bestRefusalSignals 同口径）。

    判据因此是：**原查询与所有改写都看不出相关性时才拒答** ——
    这正好补上词表信号在语义化提问上的失效（一条术语化改写命中语料，
    就说明文档里确实有这件事，不该粗拒）。
    """
    max_cosine: float | None = None
    cos_gap: float | None = None
    max_bm25: float | None = None
    idf_coverage: float | None = None

    def take(cur: float | None, nxt: float | None) -> float | None:
        if nxt is None:
            return cur
        if cur is None:
            return nxt
        return max(cur, nxt)

    for variant in variants:
        signals = compute_refusal_signals(variant.bm25_scores, variant.vec_scores)
        max_cosine = take(max_cosine, signals.max_cosine)
        cos_gap = take(cos_gap, signals.cos_gap)
        max_bm25 = take(max_bm25, signals.max_bm25)
        idf_coverage = take(idf_coverage, variant.idf_coverage)
    return RefusalSignals(
        max_cosine=max_cosine, cos_gap=cos_gap, max_bm25=max_bm25, idf_coverage=idf_coverage
    )


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
    hits: list[SearchHit], limit: int, lam: float = 0.9
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


def compute_refusal_signals(
    bm25_scores: Sequence[float],
    vec_scores: Sequence[float] | None,
) -> RefusalSignals:
    """从**全量**两路原始分算出查询级拒答信号（最高余弦 / 对比度 / BM25 最高分）。

    单一实现：tune 第 7 段的信号诊断与检索管线共用本函数，保证「诊断扫出的阈值」
    与「线上门限看到的取值」逐位一致。遍历按数组原序累加（不排序），
    避免浮点加法结合律带来的均值漂移 —— 与 TS 的 `for (const v of vecScores)` 同序。
    """
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else None
    if vec_scores is None or len(vec_scores) == 0:
        return RefusalSignals(max_bm25=max_bm25)
    max_cos = float("-inf")
    total = 0.0
    for value in vec_scores:
        if value > max_cos:
            max_cos = value
        total += value
    return RefusalSignals(
        max_cosine=max_cos,
        cos_gap=max_cos - total / len(vec_scores),
        max_bm25=max_bm25,
    )


def apply_threshold(
    hits: list[SearchHit],
    options: RetrievalOptions,
    *,
    vec_available: bool,
    idf_coverage: float | None = None,
    top_k: int | None = None,
    max_cosine: float | None = None,
    cos_gap: float | None = None,
    max_bm25: float | None = None,
) -> ThresholdResult:
    """门限 + 自适应 K + 检索侧**粗拒**（精排不参与拒答判定）。

    粗拒只看**语义信号**（最高余弦 ∪ 余弦对比度）。词表信号（IDF 覆盖率 / BM25 最高分）
    不再是**一票否决**：2026-10-15 逐条诊断证明它们在 s 组与负样本上完全重叠
    （s 的 maxBm25 4.46~7.51 / IDF 0.137~0.162，负样本 4.77~7.06 / 0.152~0.197），
    放进并集只会把 s 组整体拒掉（实测 s recall@3 0.50 → 0.07），而负样本拒答仅 4/10 → 9/10。
    职责按路线图拆开：检索侧粗拒、细拒交答案侧 verifyAnswer（`parity/verify.should_refuse`）。
    词表信号仍然计算并随响应返回，供答案侧与排障使用。

    **只在向量可用时启用**：降级纯 BM25 时保持原行为（否则 BM25-only 基线自己也会拒正样本，
    相对门槛 hybrid ≥ BM25-only × 0.95 失去可比性）。
    """
    min_vec = options.min_vec_score
    ratio = options.relative_cut_ratio
    exempt_rank = options.bm25_exempt_rank
    limit = options.top_k if top_k is None else top_k

    if vec_available:
        if (max_cosine is not None and max_cosine < options.min_max_cosine) or (
            cos_gap is not None and cos_gap < options.min_cos_gap
        ):
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
    variants: Sequence[QueryVariantScores] | None = None,
) -> RetrievalResult:
    """检索后处理全流程（顺序与 TS 一致，见模块 docstring）。

    `variants`：查询扩展的额外变体。它们参与 RRF 排名与拒答信号，
    但**不改变命中上的分数字段**（见 rrf_fuse_runs）。
    """
    primary = QueryVariantScores(
        bm25_scores=bm25_scores, vec_scores=vec_scores, idf_coverage=idf_coverage
    )
    runs = [primary, *(variants or [])]
    fused = rrf_fuse_runs(
        docs,
        runs,
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

    # 拒答信号透传：idf_coverage 由调用方（检索服务/评测）从同一份 BM25 统计算出；
    # 余弦类信号用 compute_refusal_signals（与 tune 第 7 段同实现、同口径）。
    signals = (
        best_refusal_signals(runs)
        if len(runs) > 1
        else RefusalSignals(**{**compute_refusal_signals(bm25_scores, vec_scores).__dict__,
                               "idf_coverage": idf_coverage})
    )
    gated = apply_threshold(
        pool,
        options,
        vec_available=any(run.vec_scores is not None for run in runs),
        idf_coverage=idf_coverage,
        top_k=options.mmr_pool_size,
        max_cosine=signals.max_cosine,
        cos_gap=signals.cos_gap,
        max_bm25=signals.max_bm25,
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
