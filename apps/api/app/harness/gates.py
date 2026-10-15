"""CI 门槛（apps/web/scripts/eval.ts 的门槛表逐条移植）。

门槛的纪律（README 工程约定）：
- **门槛 = 实测 − 余量**，作用是防回归，不是定目标；每条都要记录取值依据与日期
- 字段名写错不许静默失效：这里用三元组 (名称, 实测值, 阈值) 而不是字符串下标取值
- 拒答类门槛必须与误拒门槛**成对**设：只压一边会把系统调成哑巴

2026-10-12 重取：修掉「评测不挂向量 → MMR 静默失效」的口径漂移后，实测值变化为
recall@3 0.7647→0.7255、q 组 0.92→0.72、s 组 0.2857→0.50、结构性冗余 0.1765→0.0980。
"""

from __future__ import annotations

from .metrics import Metrics

#: 分组 recall@3 门槛（2026-10-12 实测值往下各留 ~0.04 余量）
BY_GROUP_GATE: dict[str, float] = {
    "q": 0.68,  # 技术措辞：实测 0.72
    "s": 0.45,  # 口语化：实测 0.50
    "m": 0.9,  # 单文档多主题：实测 1.00
}

#: 上限类门槛（超过即失败）
MAX_GATES: list[tuple[str, float]] = [
    ("top3NearDuplicateRate", 0.0),   # 内容性冗余：出现一条近似重复即红
    ("falseRefusalCount", 0.0),       # 正样本被拒 = 漏检，必须为 0
]

#: 下限类门槛（低于即失败）。绝对值门槛在换语料后失效，所以另配相对门槛。
MIN_GATES: list[tuple[str, float]] = [
    ("negativeRefusedCount", 3.0),    # 10 条负样本至少拒掉 3 条（实测 4 条）
]

#: 相对门槛：hybrid recall@3 ≥ BM25-only × 0.95（与语料难度无关）
RELATIVE_HYBRID_FLOOR = 0.95

#: 校验组件 P/R/F1 门槛（calibrate-verify.ts --check 同值）
VERIFY_GATES = {"precision": 0.7, "recall": 0.9, "f1": 0.75}


def check_gates(metrics: Metrics) -> tuple[bool, list[str]]:
    """返回 (是否全部通过, 失败项说明)。"""
    failed: list[str] = []
    values = {
        "top3NearDuplicateRate": metrics.top3_near_duplicate_rate,
        "falseRefusalCount": float(metrics.false_refusal_count),
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
    if metrics.bm25_only_recall_at_3 is not None:
        floor = metrics.bm25_only_recall_at_3 * RELATIVE_HYBRID_FLOOR
        if metrics.recall_at_3 < floor:
            failed.append(
                f"recall@3({metrics.recall_at_3}) < BM25-only({metrics.bm25_only_recall_at_3})"
                f"×{RELATIVE_HYBRID_FLOOR}={floor:.4f}（向量路拖后腿）"
            )
    for group, minimum in BY_GROUP_GATE.items():
        bucket = metrics.by_group.get(group)
        if bucket is None:
            failed.append(f"缺少分组 {group}")
            continue
        if bucket["recallAt3"] < minimum:
            failed.append(f"{group}.recall@3={bucket['recallAt3']} < {minimum}")
        if bucket["noHit"] > 0:
            failed.append(f"{group}.被拒 {bucket['noHit']} 条")
    return not failed, failed
