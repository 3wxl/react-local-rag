"""幻觉后处理（apps/web/src/utils/verifyAnswer.ts 的移植）。

两路证据（不调模型自评，纯确定性校验）：
1. 语义证据：答案逐句向量化，与**实际引用的片段及其窗口**算最大余弦
2. 词法证据：中文二元组/单词覆盖率 + 数字事实核查（答案中的数字必须能在原文找到）

两路加权后给每句裁决 supported / weak / unsupported，再汇总整体裁决。
向量不可用时降级为纯词法判定（`source="lexical"`）。

语义分由调用方注入（`semantic_scorer`）：服务端用本地 ONNX，测试用固定值，
算法本身与打分来源解耦 —— 这样阈值与裁决公式可以单独测。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .citation import VerifyPassage
from .jsstr import JS_WS, js_trim
from .sentences import split_sentences
from .tokenize import tokenize, tokenize_set

SentenceLevel = str  # "supported" | "weak" | "unsupported"
VerifyVerdict = str  # "supported" | "partial" | "unsupported" | "refused"


@dataclass
class VerifyThresholds:
    """默认阈值由 packages/eval/verify-labels.json 的人工标注集校准得到。"""

    supported: float = 0.55
    weak: float = 0.38
    sem_weight: float = 0.75
    lex_weight: float = 0.25


DEFAULT_VERIFY_THRESHOLDS = VerifyThresholds()

#: 句子至少包含多少个有效字符才算"主张句"，否则视为寒暄句跳过
MIN_CLAIM_CHARS = 4

#: 检索门限过滤后一条都不剩时的标准拒答文本
NO_RELEVANT_ANSWER = "文档中没有找到相关内容。"

#: 模型按指令拒答时的标志性短语（说明模型没有编造，应判为合规拒答）
_REFUSAL_RE = re.compile(r"(没有找到|未找到|找不到|无法回答|未能找到|不含相关|没有相关)")

#: 引用角标：[1] / [1,2] / [1，2] / [1][2]
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")

#: 数字事实（阿拉伯数字，可带小数/百分号/常见单位）
_NUMBER_RE = re.compile(
    rf"\d+(?:\.\d+)?[{re.escape(JS_WS)}]*(?:%|％|元|块|天|小时|分钟|个|折|倍|年|月|周|页|万|亿)?"
)
_WS_RE = re.compile(f"[{re.escape(JS_WS)}]+")
_NUMBER_CORE_RE = re.compile(r"\d+(?:\.\d+)?")

#: isClaimSentence 里显式排除的符号（\p{P}\p{S} 之外再保险一份，含中文标点）
_PUNCT_EXTRA = set("。！？!?；;，,、：:＂\"＇'（）()【】[]-—…·")


@dataclass
class VerifiedSentence:
    text: str
    level: SentenceLevel
    score: float
    semantic: float | None
    lexical: float
    unknown_numbers: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    #: 主张句但没有**有效**角标（"根本没引"，区别于 invalid_citations 的"引错了"）
    uncited: bool = False
    trivial: bool = False


@dataclass
class CitationConvergence:
    """引用闭环收敛情况（逐句角标 → 越界检测 → 证据收敛）。"""

    uncited_claims: int = 0
    out_of_range_claims: int = 0
    citations_subset_of_passages: bool = True


@dataclass
class VerificationResult:
    verdict: VerifyVerdict
    sentences: list[VerifiedSentence]
    source: str  # semantic | lexical
    scoped_chunk_count: int = 0
    convergence: CitationConvergence = field(default_factory=CitationConvergence)


# ---------------- 打分与裁决 ----------------


def compute_sentence_score(
    semantic: float | None,
    lexical: float,
    thresholds: VerifyThresholds = DEFAULT_VERIFY_THRESHOLDS,
    *,
    fabricated_number: bool = False,
    dangling_citation: bool = False,
    uncited_claim: bool = False,
) -> float:
    """单句依据分（三个硬信号力度不同，见 TS 注释）。

    - 数字捏造且语义依据不强（<0.6）→ 压到 weak 之下（判 unsupported，红色告警）
    - 编造引用 → 一律封顶到 weak（黄色告警）：句子本身可能真，但角标对不上原文
    - 无角标主张句 → 同样封顶到 weak：句子可能完全为真，但**无法回跳核对**
    """
    score = (
        semantic * thresholds.sem_weight + lexical * thresholds.lex_weight
        if semantic is not None
        else lexical
    )
    if fabricated_number and (semantic is None or semantic < 0.6):
        score = min(score, thresholds.weak - 0.01)
    if dangling_citation:
        score = min(score, thresholds.weak)
    if uncited_claim:
        score = min(score, thresholds.weak)
    return score


def level_of(
    score: float, thresholds: VerifyThresholds = DEFAULT_VERIFY_THRESHOLDS
) -> SentenceLevel:
    if score >= thresholds.supported:
        return "supported"
    if score >= thresholds.weak:
        return "weak"
    return "unsupported"


def strip_think_tags(text: str) -> str:
    """去除模型思考标签（标签用拼接构造，避免源码里出现整块标签被工具改写）。"""
    open_tag = "<" + "think" + ">"
    close_tag = "<" + "/think" + ">"
    out = text
    i = out.find(open_tag)
    if i != -1:
        j = out.find(close_tag, i + len(open_tag))
        out = out[:i] + out[j + len(close_tag) :] if j != -1 else out[:i]
    return js_trim(out.replace(open_tag, "").replace(close_tag, ""))


def _scan_citations(text: str, minimum: int) -> list[int]:
    """扫描角标编号：minimum=1 只取合法编号；minimum=0 连 [0] 一起取（越界检测用）。"""
    out: set[int] = set()
    for m in _CITATION_RE.finditer(text):
        for part in re.split(r"[,，]", m.group(1)):
            n = int(part.strip())
            if n >= minimum:
                out.add(n)
    return sorted(out)


def parse_citations(text: str) -> list[int]:
    """解析句子中的引用编号（去重升序）。"""
    return _scan_citations(text, 1)


def parse_all_citations(text: str) -> list[int]:
    """解析**全部**角标编号（含 [0] 这类永远不合法的编号）。

    旧实现用 `n > 0` 过滤，`[0]` 被静默丢弃当作"无引用"，随后走 fallback 范围
    比语义分 —— 反而最容易判成 supported。编造引用不能因为编号难看就装作没看见。
    """
    return _scan_citations(text, 0)


def should_refuse(result: VerificationResult) -> bool:
    """拒答裁决（答案侧细拒，M2）——与 TS shouldRefuse 同一套规则。

    职责划分：检索侧只做**粗拒**（明显的无关查询），「文档里有没有这件事」的细判归这里，
    依据是逐句证据（语义 + 词法 + 数字核查）。三条判据：
    1. verdict == "refused"：模型自己按指令拒答，合规
    2. verdict == "unsupported"：全部主张句都无依据
    3. source == "lexical" 且 verdict != "supported"：只剩词法证据时宁严勿宽

    刻意**不**因为 partial 就拒答：部分句子弱依据（含"无角标"这类格式问题）
    不等于答案无价值，那属于 UI 告警层。拒答越激进，正样本误拒越多。
    """
    if result.verdict == "refused":
        return True
    if result.verdict == "unsupported":
        return True
    if result.source == "lexical" and result.verdict != "supported":
        return True
    return False


def summarize_verdict(sentences: Sequence[VerifiedSentence]) -> VerifyVerdict:
    """整体裁决（只统计主张句）——与 TS summarizeVerdict 同一套聚合规则。"""
    claims = [s for s in sentences if not s.trivial]
    if not claims:
        return "supported"
    unsupported = [s for s in claims if s.level == "unsupported"]
    weak = [s for s in claims if s.level == "weak"]
    if len(unsupported) == len(claims):
        return "unsupported"
    if unsupported or weak:
        return "partial"
    return "supported"


# ---------------- 词法层 ----------------


def extract_numbers(text: str) -> list[str]:
    """抽取数字事实（去掉数字与单位之间的空白，与 TS 一致）。"""
    return [_WS_RE.sub("", m.group(0)) for m in _NUMBER_RE.finditer(text)]


def number_core(matched: str) -> str:
    """从带单位的数字串里取出纯数字核（"15天" → "15"）。"""
    m = _NUMBER_CORE_RE.match(matched)
    return m.group(0) if m else matched


def context_contains_number(core: str, context_text: str) -> bool:
    """上下文里是否出现某个数字核（带数字边界，避免 "100" 命中 "1000"）。"""
    escaped = re.escape(core)
    return re.search(rf"(?<!\d){escaped}(?!\d)", context_text) is not None


def find_unknown_numbers(
    sentence: str,
    context_text: str,
    context_numbers: set[str] | None = None,
) -> list[str]:
    """找出句子中「原文里查无出处」的数字（数字捏造信号）。

    引用角标不是数字事实：先剥掉 [1]/[1,2] 再抽取。否则角标编号会被当成
    「捏造的数字」（语料里未必有孤立的"1"），无向量环境（纯词法）下
    fabricated_number 封顶必生效 → 带引用的正确答案被误拒（CI 必现）。
    """
    nums = context_numbers if context_numbers is not None else set(extract_numbers(context_text))
    facts = extract_numbers(_CITATION_RE.sub("", sentence))
    return [
        n
        for n in facts
        if n not in nums and not context_contains_number(number_core(n), context_text)
    ]


def lexical_coverage(sentence: str, context_tokens: set[str]) -> float:
    """词法覆盖率：句子的有效 token 中有多少能在上下文 token 集合里找到。"""
    tokens = tokenize(sentence)
    if not tokens:
        return 1.0
    unique = set(tokens)
    hit = sum(1 for t in unique if t in context_tokens)
    return hit / len(unique)


def is_claim_sentence(sentence: str) -> bool:
    """判断句子是否为主张句（去掉空白/标点/符号后仍 ≥ MIN_CLAIM_CHARS 个字符）。

    TS 用 `/\\p{P}\\p{S}/u`，Python `re` 不支持 `\\p{...}`；
    这里用 `unicodedata.category` 精确还原（P* 标点、S* 符号），不引入第三方 regex 依赖。
    """
    core = [
        ch
        for ch in sentence
        if not ch.isspace()
        and ch not in _PUNCT_EXTRA
        and unicodedata.category(ch)[0] not in ("P", "S")
    ]
    return len(core) >= MIN_CLAIM_CHARS


# ---------------- 校验范围 ----------------


def _scope_for(citations: Sequence[int], passages: Sequence[VerifyPassage]) -> list[str]:
    """按引用编号取校验范围（块 id 集合）；无引用则用全部段落。"""
    all_ids = [cid for p in passages for cid in p.chunk_ids]
    if not citations:
        return all_ids
    by_index = {p.index: p for p in passages}
    scoped = [
        cid for c in citations for cid in (by_index[c].chunk_ids if c in by_index else [])
    ]
    return scoped if scoped else all_ids


def build_refusal_result(answer: str) -> VerificationResult:
    """构造拒答的校验结果（全部句子 trivial，不得触发依赖告警）。"""
    return VerificationResult(
        verdict="refused",
        source="lexical",
        scoped_chunk_count=0,
        sentences=[
            VerifiedSentence(
                text=text,
                level="supported",
                score=1.0,
                semantic=None,
                lexical=1.0,
                trivial=True,
            )
            for text in split_sentences(answer)
        ],
    )


def is_invalid_citation(n: int, passage_indexes: set[int], has_passages: bool) -> bool:
    """引用编号合法性判据（**集合判定**，不是"≤ 最大值"）。

    旧实现用 `c > max_passage_index` 判越界，有两个洞：`[0]` 被 parse_citations 的
    `n > 0` 提前吞掉（永远检测不到）；编号段不连续时（1,3）引用中间的 [2] 会被当成合法。
    passages 为空（旧调用：没有编号可引）时沿用旧口径（正编号一律视为越界），行为不变。
    """
    if not has_passages:
        return n > 0
    return n <= 0 or n not in passage_indexes


def verify_answer(
    answer: str,
    context_text: str,
    *,
    passages: Sequence[VerifyPassage] = (),
    thresholds: VerifyThresholds = DEFAULT_VERIFY_THRESHOLDS,
    semantic_scorer: Callable[[list[str], list[str] | None], list[float]] | None = None,
    require_citation: bool = True,
) -> VerificationResult:
    """对模型最终答案做事后校验。

    @param answer 模型最终答案（调用前建议先 strip_think_tags）
    @param context_text 喂给模型的上下文文本（词法覆盖率的分母）
    @param passages 编号段落表（语义范围收敛）；为空则对全文上下文校验
    @param semantic_scorer 语义打分回调 (sentences, scope_chunk_ids) -> 每句最大余弦；
                            为 None 或抛错时降级纯词法（对应 TS 里 worker 不可用）
    @param require_citation 是否强制逐句角标（默认 True；仅在提供 passages 时生效）。False 只用于对照测量
    """
    has_passages = len(passages) > 0
    passage_indexes = {p.index for p in passages}
    segments = split_sentences(answer)

    # 模型按指令拒答：答案短且含拒答短语，判为合规，不算幻觉
    compact = _WS_RE.sub("", answer)
    refusal_phrases = (
        "没有找到",
        "未找到",
        "找不到",
        "无法回答",
        "未能找到",
        "不含相关",
        "没有相关",
    )
    if len(compact) <= 80 and any(p in answer for p in refusal_phrases):
        return build_refusal_result(answer)

    # 只把主张句送进语义比对，保持顺序对齐
    claim_flags = [is_claim_sentence(s) for s in segments]
    # citations 保持"正编号"语义（契约不变）；越界检测另用含 [0] 的全量扫描
    cited_per_sentence = [parse_citations(s) for s in segments]
    invalid_per_sentence = [
        [
            c
            for c in parse_all_citations(text)
            if is_invalid_citation(c, passage_indexes, has_passages)
        ]
        for text in segments
    ]
    # 有编号段落表时，主张句必须带**有效**角标（引用强制闭环）
    require = has_passages and require_citation
    uncited_per_sentence = [
        bool(
            require
            and claim_flags[idx]
            and not any(c in passage_indexes for c in cited_per_sentence[idx])
        )
        for idx in range(len(segments))
    ]

    context_tokens = tokenize_set(context_text)
    context_numbers = set(extract_numbers(context_text))

    # 按「引用范围」分组：同一范围的句子合并成一次打分调用
    def scope_key(citations: list[int]) -> str:
        if not has_passages:
            return "all"
        if not citations:
            return "fallback"
        return ",".join(str(c) for c in sorted(citations))

    groups: dict[str, dict] = {}
    for idx, text in enumerate(segments):
        if not claim_flags[idx]:
            continue
        # 越界引用不参与范围计算（视为无引用，走 fallback 范围）
        valid = [c for c in cited_per_sentence[idx] if c in passage_indexes]
        key = scope_key(valid)
        scope = _scope_for(valid, passages)
        if key in groups:
            groups[key]["texts"].append(text)
        else:
            groups[key] = {"texts": [text], "scope": scope}

    semantic_map: dict[str, float] = {}
    source = "semantic"
    scoped_chunk_count = 0
    try:
        if semantic_scorer is None:
            raise RuntimeError("语义打分器不可用")
        for group in groups.values():
            scoped_chunk_count = max(scoped_chunk_count, len(group["scope"]))
            scores = semantic_scorer(
                group["texts"], group["scope"] if passages else None
            )
            for text, score in zip(group["texts"], scores):
                semantic_map[text] = score
    except Exception:  # noqa: BLE001 —— 与 TS 一致：语义不可用不阻塞回答展示
        source = "lexical"

    sentences: list[VerifiedSentence] = []
    for idx, text in enumerate(segments):
        lexical = lexical_coverage(text, context_tokens)
        unknown_numbers = find_unknown_numbers(text, context_text, context_numbers)

        if not claim_flags[idx]:
            # 寒暄/语气句不参与裁决
            sentences.append(
                VerifiedSentence(
                    text=text,
                    level="supported",
                    score=1.0,
                    semantic=None,
                    lexical=lexical,
                    citations=cited_per_sentence[idx],
                    invalid_citations=invalid_per_sentence[idx],
                    trivial=True,
                )
            )
            continue

        semantic = semantic_map.get(text) if source == "semantic" else None
        score = compute_sentence_score(
            semantic,
            lexical,
            thresholds,
            fabricated_number=bool(unknown_numbers),
            dangling_citation=bool(invalid_per_sentence[idx]),
            uncited_claim=uncited_per_sentence[idx],
        )
        sentences.append(
            VerifiedSentence(
                text=text,
                level=level_of(score, thresholds),
                score=score,
                semantic=semantic,
                lexical=lexical,
                unknown_numbers=unknown_numbers,
                citations=cited_per_sentence[idx],
                invalid_citations=invalid_per_sentence[idx],
                uncited=uncited_per_sentence[idx],
                trivial=False,
            )
        )

    claims = [s for s in sentences if not s.trivial]
    out_of_range_claims = sum(1 for s in claims if s.invalid_citations)
    convergence = CitationConvergence(
        uncited_claims=sum(1 for s in claims if s.uncited),
        out_of_range_claims=out_of_range_claims,
        # 收敛判据：不存在任何越界角标（每个引用编号都能在编号段落表里找到）
        citations_subset_of_passages=out_of_range_claims == 0,
    )

    return VerificationResult(
        verdict=summarize_verdict(sentences),
        sentences=sentences,
        source=source,
        scoped_chunk_count=scoped_chunk_count,
        convergence=convergence,
    )
