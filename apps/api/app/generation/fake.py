"""确定性假生成通道（测试 / CI 用）。

不下载权重、不依赖服务，但**必须遵守与真模型同样的契约**：
- 有文档依据时逐句带 [n] 角标（否则校验链路测不到东西）
- 无依据时输出标准拒答文本
- 遇到上下文里的数字要原样引用（数字核查才不会假阳性）

它存在的意义：把「服务端问答链路」（检索 → prompt → 生成 → 校验 → 引用 → 落库）
从「模型是否可用」里解耦出来，让 API 契约测试可以在无模型环境跑。
"""

from __future__ import annotations

from .base import Completion


class FakeProvider:
    name = "fake"
    model = "fake-cited-answer"

    def __init__(self, *, latency_ms: int = 0) -> None:
        self.latency_ms = latency_ms

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
        """从 Prompt 里读出编号段落，逐段回抄一句并标注角标（模拟"照着文档答"）。"""
        passages = _parse_passages(prompt)
        if not passages:
            text = "文档中没有找到相关内容。"
        else:
            lines = []
            for index, body in passages:
                first = _first_sentence(body)
                if first:
                    lines.append(f"{first}[{index}]")
            text = "".join(lines) if lines else "文档中没有找到相关内容。"
        return Completion(
            text=text,
            prompt_tokens=len(prompt),
            completion_tokens=len(text),
            model=self.model,
            provider=self.name,
            latency_ms=self.latency_ms,
            sampling={"deterministic": True, "max_new_tokens": max_new_tokens},
        )


def _parse_passages(prompt: str) -> list[tuple[int, str]]:
    """从 Prompt 的【文档内容】段里解析出 `[n] label\\n正文`（与 prompt 模板同构）。"""
    marker = "【文档内容】\n"
    start = prompt.find(marker)
    if start < 0:
        return []
    body = prompt[start + len(marker) :]
    end = body.find("\n<|im_end|>")
    if end >= 0:
        body = body[:end]
    out: list[tuple[int, str]] = []
    for block in body.split("\n\n"):
        if not block.startswith("["):
            continue
        head, _, rest = block.partition("\n")
        try:
            index = int(head[1 : head.index("]")])
        except ValueError:
            continue
        out.append((index, rest.strip()))
    return out


def _first_sentence(text: str) -> str:
    """取第一句（保留标点），与分块器的句子切分口径同源。"""
    from ..parity.sentences import split_sentences_keep

    parts = split_sentences_keep(text)
    return parts[0].strip() if parts else ""
