"""内置工具（M3）：search_docs / doc_outline / read_chunks / calculator。

与浏览器端 `apps/web/src/agent/toolAgent.ts` 的工具体系**同语义、不同实现位置**：
那边的工具跑在浏览器（IndexedDB 索引），这边的跑在服务端（store + 检索服务）。
语义对齐的意义是"同一个问题两种模式给出同族结果"，不是字节级一致（工具输出本来
就不进契约，只有最终的 answer/citations 进）。

权限：检索类工具一律带 ACL Filters（与 /v1/retrieve、/v1/answer 同一条召回路径）——
工具面不能成为绕过权限的旁路。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from ...parity.citation import build_evidence
from ...store import Filters
from .registry import ToolOutcome, ToolSpec

if TYPE_CHECKING:
    from ...api.serializers import context_to_dict  # noqa: F401 —— 仅供类型阅读
    from ...core.config import Settings
    from ...embedding.bge import BgeOnnxEmbedder
    from ...services.retrieval_service import RetrievalService
    from ...store import Store

#: 工具返回给模型/轨迹的片段长度上限（防止一次工具调用把上下文撑爆）
MAX_SNIPPET_CHARS = 400


def _snippet(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def build_builtin_tools(
    *,
    store: "Store",
    retrieval: "RetrievalService",
    embedder: "BgeOnnxEmbedder",
    settings: "Settings",
    version_getter: Callable[[], str],
    filters_getter: Callable[[], Filters | None],
) -> list[ToolSpec]:
    """构造内置工具。

    `version_getter` / `filters_getter` 用取值函数而不是值：工具调用发生在一次 run 内，
    版本与 ACL 过滤在 run 开始时确定（由路由解析），取值函数让节点不必把这些塞进 State。
    """
    from ...api.serializers import context_to_dict

    async def search_docs(args: dict) -> ToolOutcome:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("search_docs 需要 query 参数")
        top_k = int(args.get("top_k") or settings.graph_tool_top_k)
        search = await retrieval.search(
            query,
            version=version_getter(),
            top_k=max(1, min(top_k, 10)),
            filters=filters_getter(),
        )
        passages, contexts = build_evidence(search.hits)
        if not search.hits:
            return ToolOutcome(
                text=f"未命中任何片段（no_relevant_hit={search.no_relevant_hit}）",
                data={
                    "no_relevant_hit": search.no_relevant_hit,
                    "evidence": [],
                    "citations": [],
                },
            )
        lines = [
            f"命中 {len(search.hits)} 条（no_relevant_hit={search.no_relevant_hit}）："
        ]
        for passage, context in zip(passages, contexts):
            lines.append(f"[{passage.index}] {passage.label}\n{_snippet(passage.text)}")
        return ToolOutcome(
            text="\n".join(lines),
            data={
                "no_relevant_hit": search.no_relevant_hit,
                "evidence": [
                    {"index": p.index, "label": p.label, "text": p.text} for p in passages
                ],
                "citations": [context_to_dict(c) for c in contexts],
            },
        )

    async def doc_outline(args: dict) -> ToolOutcome:
        doc_name = str(args.get("doc_name") or "").strip()
        chunks = await store.load_chunks(version_getter(), filters=filters_getter())
        docs = await store.list_documents()
        targets = [d for d in docs if (not doc_name or d.doc_name == doc_name)]
        if not targets:
            return ToolOutcome(text=f"没有文档：{doc_name or '（知识库为空）'}")
        lines: list[str] = []
        for doc in targets:
            own = [c for c in chunks if c.doc_id == doc.doc_id]
            headings: list[str] = []
            for chunk in sorted(own, key=lambda c: c.ord):
                path = [p for p in (chunk.heading_path or []) if p]
                label = " · ".join(path) if path else "正文"
                if label not in headings:
                    headings.append(label)
            lines.append(
                f"{doc.doc_name}（{len(own)} 块）：\n"
                + "\n".join(f"  - {h}" for h in headings[:20])
            )
        return ToolOutcome(text="\n".join(lines))

    async def read_chunks(args: dict) -> ToolOutcome:
        wanted = [str(cid) for cid in (args.get("chunk_ids") or [])]
        if not wanted:
            raise ValueError("read_chunks 需要 chunk_ids 参数")
        chunks = await store.load_chunks(version_getter(), filters=filters_getter())
        by_id = {c.chunk_id: c for c in chunks}
        found = [by_id[cid] for cid in wanted if cid in by_id]
        if not found:
            return ToolOutcome(text="这些块不存在或无权访问（chunk_ids 可能来自上一版本索引）")
        text = "\n\n".join(f"[{c.chunk_id}] {_snippet(c.content)}" for c in found)
        return ToolOutcome(text=text)

    async def calculator(args: dict) -> ToolOutcome:
        expression = str(args.get("expression") or "")
        result = evaluate_expression(expression)
        if not result["ok"]:
            # 计算失败也是"观察"：让模型自己改成合法算式，而不是把整轮问答应答打断
            raise ValueError(str(result["error"]))
        return ToolOutcome(text=f"{expression} = {result['formatted']}")

    return [
        ToolSpec(
            name="search_docs",
            description="在知识库中检索片段（带引用标签与编号），返回命中文本。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "top_k": {"type": "integer", "description": "返回条数（1-10）"},
                },
                "required": ["query"],
            },
            handler=search_docs,
        ),
        ToolSpec(
            name="doc_outline",
            description="列出文档的章节大纲（不传 doc_name 时列出全部文档）。",
            parameters={
                "type": "object",
                "properties": {"doc_name": {"type": "string"}},
            },
            handler=doc_outline,
        ),
        ToolSpec(
            name="read_chunks",
            description="按 chunk_ids 读取块的原文（用于核对检索片段的上下文）。",
            parameters={
                "type": "object",
                "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["chunk_ids"],
            },
            handler=read_chunks,
        ),
        ToolSpec(
            name="calculator",
            description="算术求值（+-*/%^ 与括号），用于文档里的数字计算。",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=calculator,
        ),
    ]


# ---------------- 计算器（apps/web/src/utils/calculator.ts 的移植） ----------------


MAX_EXPRESSION_CHARS = 200


def evaluate_expression(input_text: str) -> dict:
    """递归下降求值器（与 TS 版同语法、同错误语义）。

    为什么不用 eval：表达式来自模型输出，属于不可信输入；这个解析器只认识数字与
    六个运算符，其它字符一律报错，攻击面为零。
    语法：expr := term (('+'|'-') term)* ；term := unary (('*'|'/'|'%') unary)* ；
         unary := ('+'|'-') unary | power ；power := primary ('^' unary)?（右结合）。
    """
    expr = (input_text or "").strip()
    if not expr:
        return {"ok": False, "error": "表达式为空"}
    if len(expr) > MAX_EXPRESSION_CHARS:
        return {"ok": False, "error": f"表达式过长（上限 {MAX_EXPRESSION_CHARS} 字符）"}
    parser = _Parser(expr)
    try:
        value = parser.parse_expression()
        parser.expect_end()
    except _ParseError as err:
        return {"ok": False, "error": str(err)}
    except ZeroDivisionError:
        return {"ok": False, "error": "除以 0（结果不是有限数）"}
    if value != value or value in (float("inf"), float("-inf")):
        return {"ok": False, "error": "结果不是有限数（可能除以 0 或溢出）"}
    return {"ok": True, "value": value, "formatted": format_number(value)}


def format_number(value: float) -> str:
    """数值格式化：消掉浮点噪声（0.1+0.2 → 0.3），保留至多 12 位有效数字。"""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12g}"


class _ParseError(Exception):
    pass


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse_expression(self) -> float:
        value = self._parse_term()
        while True:
            self._skip_spaces()
            if self._peek() in ("+", "-"):
                op = self._next()
                rhs = self._parse_term()
                value = value + rhs if op == "+" else value - rhs
            else:
                return value

    def _parse_term(self) -> float:
        value = self._parse_unary()
        while True:
            self._skip_spaces()
            if self._peek() in ("*", "/", "%"):
                op = self._next()
                rhs = self._parse_unary()
                if op == "*":
                    value = value * rhs
                elif op == "/":
                    if rhs == 0:
                        raise ZeroDivisionError
                    value = value / rhs
                else:
                    if rhs == 0:
                        raise ZeroDivisionError
                    # 取模（与 JS 的 % 语义一致：结果符号跟随被除数）
                    value = value - int(value / rhs) * rhs
            else:
                return value

    def _parse_unary(self) -> float:
        self._skip_spaces()
        if self._peek() in ("+", "-"):
            op = self._next()
            value = self._parse_unary()
            return -value if op == "-" else value
        return self._parse_power()

    def _parse_power(self) -> float:
        base = self._parse_primary()
        self._skip_spaces()
        if self._peek() == "^":
            self._next()
            exponent = self._parse_unary()  # 右结合：2^3^2 = 2^(3^2)
            return float(base**exponent)
        return base

    def _parse_primary(self) -> float:
        self._skip_spaces()
        char = self._peek()
        if char == "(":
            self._next()
            value = self.parse_expression()
            self._skip_spaces()
            if self._peek() != ")":
                raise _ParseError("括号不匹配")
            self._next()
            return value
        return self._parse_number()

    def _parse_number(self) -> float:
        self._skip_spaces()
        start = self.pos
        while self._peek().isdigit() or self._peek() == ".":
            self.pos += 1
        if start == self.pos:
            raise _ParseError(f"位置 {self.pos} 处不是数字或括号：{self.text!r}")
        try:
            return float(self.text[start : self.pos])
        except ValueError as err:
            raise _ParseError(f"无法解析数字：{self.text[start : self.pos]!r}") from err

    def expect_end(self) -> None:
        self._skip_spaces()
        if self.pos != len(self.text):
            raise _ParseError(f"位置 {self.pos} 处有多余字符：{self.text[self.pos]!r}")

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _next(self) -> str:
        char = self._peek()
        self.pos += 1
        return char

    def _skip_spaces(self) -> None:
        while self._peek() and self._peek().isspace():
            self.pos += 1
