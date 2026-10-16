"""查询扩展 Prompt（apps/web/src/utils/expandPrompt.ts 的移植，**逐字节相同**）。

一端改一个字，双端的改写就会不同 —— 与 answer_prompt.py 同一条纪律，
由 tests/test_prompt_verify_parity.py 的字节级断言钉住。

任务定位：把**口语化提问**改写成**技术文档里可能出现的检索关键词**。
为什么是"关键词"而不是"改写句子"：0.5B 模型做"抽取"远比做"改写"可靠 ——
让它改写句子时输出的是冗长复述（仍不含语料术语，词表信号照样是 0），
改成关键词后同一模型给出的词组开始命中语料（详见 TS 侧注释的实测对比）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExpandPromptInput:
    question: str
    max: int = 3


def build_expand_prompt(input: ExpandPromptInput) -> str:
    return (
        "<|im_start|>system\n"
        "你是检索助手。把用户问题改写成**技术文档里可能出现的检索关键词**。\n"
        "\n"
        "规则：\n"
        "1. 只输出名词或术语，不要成句、不要疑问词\n"
        "2. 每行一组，词与词之间用空格分隔\n"
        "3. 不要添加原问题里没有的信息\n"
        f"4. 最多 {input.max} 行，每行最多 10 个词<|im_end|>\n"
        "<|im_start|>user\n"
        f"{input.question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
