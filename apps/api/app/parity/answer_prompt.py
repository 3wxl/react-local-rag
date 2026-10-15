"""问答 Prompt 构造（apps/web/src/utils/answerPrompt.ts 的逐字节移植）。

这个模板必须与 TS 完全一致（含空格与换行）：prompt 差一个字符，模型行为就可能不同，
双端对比也就失去意义。测试用 UTF-8 字节级比对钉住（tests/test_prompt_parity.py）。

要求模型先输出 `<think>` 思考再给答案，并逐句标注引用角标 [n]；
角标是后续「校验范围收敛」与「引用卡片」的对接点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PromptPassage:
    """喂给模型的编号段落。"""

    index: int
    label: str
    text: str


@dataclass
class HistoryTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class AnswerPromptInput:
    question: str
    passages: Sequence[PromptPassage] = field(default_factory=list)
    history: Sequence[HistoryTurn] = ()
    history_summary: str | None = None


def build_answer_prompt(inp: AnswerPromptInput) -> str:
    """构造 Qwen chat 格式的完整 Prompt（与 TS buildAnswerPrompt 同字节）。"""
    question = inp.question
    passages = inp.passages
    history = inp.history
    history_summary = inp.history_summary

    # 编号段落：每段带 [n] 角标 +「文件名 · 章节 · 页码」标签
    context = (
        "\n\n".join(f"[{p.index}] {p.label}\n{p.text}" for p in passages)
        if passages
        else ""
    )
    if passages:
        citation_rule = (
            "引用规则（必须遵守）：\n"
            "1. 每句话结尾都要标注依据编号，格式 [1]；一句话用了多段依据就写 [1][2]\n"
            "2. 只能引用【文档内容】中列出的编号，禁止编造编号\n"
            "3. 文档中没有依据的内容一律不写；整篇都没有依据时只回复：文档中没有找到相关内容\n"
        )
    else:
        citation_rule = "文档无相关内容，直接回复：文档中没有找到相关内容，禁止编造信息。\n"

    history_block = "\n".join(
        f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>" for turn in history
    )

    return (
        "<|im_start|>system\n"
        "你是文档问答助手，只能使用【文档内容】回答。\n"
        "回答前先写出简短思考，再输出答案。\n"
        f"{citation_rule}"
        '可结合【此前对话摘要】与最近对话理解用户追问中的指代（如"它""上面提到的"），'
        "但回答依据仍必须来自文档内容。\n"
        "【文档内容】\n"
        f"{context}"
        f"{chr(10) + '【此前对话摘要】' + chr(10) + history_summary if history_summary else ''}\n"
        "<|im_end|>\n"
        f"{history_block + chr(10) if history_block else ''}"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
