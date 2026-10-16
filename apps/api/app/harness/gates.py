"""CI 门槛（apps/web/scripts/eval.ts 的门槛表逐条移植）。

门槛的纪律（README 工程约定）：
- **门槛 = 实测 − 余量**，作用是防回归，不是定目标；每条都要记录取值依据与日期
- 字段名写错不许静默失效：这里用三元组 (名称, 实测值, 阈值) 而不是字符串下标取值
- 拒答类门槛必须与误拒门槛**成对**设：只压一边会把系统调成哑巴

2026-10-12 重取：修掉「评测不挂向量 → MMR 静默失效」的口径漂移后，实测值变化为
recall@3 0.7647→0.7255、q 组 0.92→0.72、s 组 0.2857→0.50、结构性冗余 0.1765→0.0980。

2026-10-15 终取：检索侧改为**语义粗拒**（词表信号降为提示）+ 查询扩展 + 关闭 MMR
（依据见 parity/retrieval.py 的 min_max_cosine / mmr_lambda 注释）。三组实测 q 0.92 / s 0.50 / m 1.00，
s 组从 0.07 回到 0.50 —— 路线图 M2「口语化 ≥0.50」达成，s 恢复为执行项。
"""

from __future__ import annotations

from .metrics import Metrics

#: 分组 recall@3 门槛（2026-10-15 实测值往下留余量）
BY_GROUP_GATE: dict[str, float] = {
    "q": 0.88,  # 技术措辞：实测 0.92（语义粗拒 + 查询扩展 + 关闭 MMR）
    "s": 0.45,  # 口语化：实测 0.50 —— 路线图 M2 目标「s ≥0.50」达成，门槛留 0.05 余量
    "m": 0.9,  # 单文档多主题：实测 1.00
}

#: 只观察、不计门槛的分组（当前为空：s 已从观察项恢复为执行项）
OBSERVED_GROUPS: frozenset[str] = frozenset()

#: 上限类门槛（超过即失败）
MAX_GATES: list[tuple[str, float]] = [
    ("top3NearDuplicateRate", 0.0),  # 内容性冗余：出现一条近似重复即红
    # 正样本被拒 = 漏检。口径：只卡**受保护人群（q+m）**—— 2026-10-15 起检索侧按
    # 「调参口径」启用融合拒答，s 被刻意降为观察项、部分会被粗拒。
    # 全量 falseRefusalCount 保留为观察值，不进门槛。
    ("protectedFalseRefusalCount", 0.0),
]

#: 下限类门槛（低于即失败）。绝对值门槛在换语料后失效，所以另配相对门槛。
#: 用**绝对条数**而不是比率：负样本只有 10 条，比率门槛等于「实测值即门槛」，
#: 翻转 1 条就掉 10pt。10 条负样本里至少拒掉 6 条（2026-10-15 实测 9/10，留 3 条余量）。
MIN_GATES: list[tuple[str, float]] = [
    # 检索侧**粗拒**（语义信号）实测拒 5/10，门槛取 4/10：只用来防"粗拒整个失效"。
    # 合计拒答率（检索 + 答案侧）由 `npm run eval:verdict` 量化，那里的门槛才是 M2 的口径。
    ("negativeRefusedCount", 4.0),
]

#: 相对门槛：hybrid recall@3 ≥ BM25-only × 0.95（与语料难度无关）
RELATIVE_HYBRID_FLOOR = 0.95

#: 校验组件 P/R/F1 门槛（calibrate-verify.ts --check 同值）
VERIFY_GATES = {"precision": 0.7, "recall": 0.9, "f1": 0.75}

#: 答案侧拒答裁决门槛（eval-verdict.ts 与 packages/eval/verdict-labels.json 的 gate 同值）。
#: 2026-10-15 定值：recall 与路线图「拒答 ≥0.60」同口径（检索粗拒 + 答案细拒合计能力）；
#: 误拒必须为 0 —— 正样本被拒是漏检，比漏拒更糟（文档明明有答案却说找不到）。
VERDICT_GATES = {"negativeRefusalRecall": 0.6, "positiveFalseRefusal": 0}


def check_gates(metrics: Metrics) -> tuple[bool, list[str]]:
    """返回 (是否全部通过, 失败项说明)。"""
    failed: list[str] = []
    values = {
        "top3NearDuplicateRate": metrics.top3_near_duplicate_rate,
        "falseRefusalCount": float(metrics.false_refusal_count),
        "protectedFalseRefusalCount": float(metrics.protected_false_refusal_count),
        "negativeRefusedCount": float(metrics.negative_refused_count),
        "recallAt3": metrics.recall_at_3,
    }
    for name, threshold in MAX_GATES:
        if name not in values:
            failed.append(f"缺少指标 {name}")
            continue
        if values[name] > threshold:
            failed.append(f"{name}={values[name]} > {threshold}")
    for name, threshold in MIN_GATES:
        if name not in values:
            failed.append(f"缺少指标 {name}")
            continue
        if values[name] < threshold:
            failed.append(f"{name}={values[name]} < {threshold}")
    if not metrics.hybrid:
        # 无向量路时不设召回/拒答类门槛（与 TS 一致：缺向量时指标本就偏保守）
        failed.append("无向量路：只有 BM25，无法判定 hybrid 门槛")
        return False, failed
    # 相对门槛只比**受保护人群（q+m）**：s 已被检索侧按调参口径刻意粗拒，
    # 纳入分母等于用「已知要放弃的人群」去评判向量质量，门槛要么恒过要么恒红，失去诊断力
    if metrics.protected_bm25_only_recall_at_3 is not None:
        floor = metrics.protected_bm25_only_recall_at_3 * RELATIVE_HYBRID_FLOOR
        if metrics.protected_recall_at_3 < floor:
            failed.append(
                f"受保护人群 recall@3({metrics.protected_recall_at_3}) < "
                f"BM25-only({metrics.protected_bm25_only_recall_at_3})"
                f"×{RELATIVE_HYBRID_FLOOR}={floor:.4f}（向量路拖后腿）"
            )
    for group, minimum in BY_GROUP_GATE.items():
        bucket = metrics.by_group.get(group)
        if bucket is None:
            failed.append(f"缺少分组 {group}")
            continue
        if group in OBSERVED_GROUPS:
            # s 只观察：它被粗拒是「拒答 ≥0.60」的已知代价，不参与门槛判定
            continue
        if bucket["recallAt3"] < minimum:
            failed.append(f"{group}.recall@3={bucket['recallAt3']} < {minimum}")
        if bucket["noHit"] > 0:
            failed.append(f"{group}.被拒 {bucket['noHit']} 条")
    return not failed, failed
