"""三级分块器（apps/web/src/utils/chunk.ts 的 Python 移植）。

这是跨语言一致性**风险最高**的一块：状态机里任何一个细节不同，
`contentHash` 就会变，索引与固化向量随之失效。移植原则是"照着抄，不做改良"。

三级：
1. 结构边界：`heading_path` 变化 → 当前缓冲整段切走（不保留重叠，组间不跨接）
2. 句子打包：攒到 token 或字符预算就切块，块尾保留整句 overlap
3. 硬切兜底：单句超过字符硬上限时按上限硬切

收尾：残块 ≥ min_chunk_chars 独立成块；否则去重叠后并入上一块（并重算 token/哈希）。

与 TS 的差异只有一处：`chunkId` 用 uuid4（TS 用 crypto.randomUUID），
id 是随机值、不参与跨端比对（见 SPEC 第 8 节）。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Union

from .chunk_params import DEFAULT_CHUNK_OPTIONS, ChunkOptions
from .jsstr import js_len, js_slice
from .fnv import fnv1a32
from .sentences import split_sentences_keep
from .token_count import count_tokens

#: 缓冲压缩阈值（与 TS PENDING_COMPACT_THRESHOLD 一致）：已消费句子数超过它就 splice 掉
PENDING_COMPACT_THRESHOLD = 256


@dataclass
class ChunkBlock:
    """解析层输出的块（对应 TS ChunkBlock / ParsedBlock）。

    `heading_path` 允许含 None：Markdown 标题层级跳跃（首个标题是 ###）时，
    TS 的 `headingPath.length = level - 1` 会造出空洞，经 JSON 往返变成 null。
    """

    text: str
    page: int = 1
    heading_path: list[str | None] = field(default_factory=list)
    type: str | None = None


@dataclass
class TextChunk:
    """带身份的结构块（对应 TS TextChunk）。字段含义见 apps/web/src/types/doc.ts。"""

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
    parent_id: str
    prev_id: str
    next_id: str
    content_hash: str


@dataclass
class ChunkStreamStats:
    pending_peak: int = 0
    compactions: int = 0
    tokens_counted: int = 0


@dataclass
class _PendingSentence:
    text: str
    start: int
    end: int
    page: int
    tokens: int


def _path_key(path: list[str | None] | None) -> str:
    """等价于 TS `JSON.stringify(block.headingPath ?? [])`（无空格、非 ASCII 不转义）。"""
    return json.dumps(
        path if path is not None else [], separators=(",", ":"), ensure_ascii=False
    )


def longest_overlap(a: str, b: str) -> int:
    """a 的末尾与 b 的开头的最长重叠长度（残块并入上一块时去重用）。"""
    for length in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:length]):
            return length
    return 0


class TextChunkStream:
    """流式分块器（逐块 push，finish 时收尾并补齐序号/链接）。"""

    def __init__(
        self,
        doc_id: str,
        doc_name: str,
        options: ChunkOptions = DEFAULT_CHUNK_OPTIONS,
    ) -> None:
        opts = options
        # 防呆校验：与 TS 一致（旧实现 overlap >= chunkSize 时悄悄钳 step=1 会切出重复尾巴）
        if opts.max_chunk_tokens <= 0:
            raise ValueError("maxChunkTokens 必须大于 0")
        if opts.min_chunk_chars >= opts.max_chunk_chars:
            raise ValueError(
                f"minChunkChars({opts.min_chunk_chars}) 必须小于 "
                f"maxChunkChars({opts.max_chunk_chars})，重叠量不可能装进一块"
            )
        if opts.sentence_overlap < 0:
            raise ValueError("sentenceOverlap 不能为负数")

        self.opts = opts
        self.doc_id = doc_id
        self.doc_name = doc_name

        self._chunks: list[TextChunk] = []
        self._pending: list[_PendingSentence] = []
        self._emit_cursor = 0
        self._pending_tokens = 0
        self._pending_chars = 0
        self._offset = 0
        self._total_chars = 0
        self._current_path_key = ""
        self._group_parent_id = ""
        self._last_chunk: TextChunk | None = None
        self.stats = ChunkStreamStats()

    # ---------------- 只读视图 ----------------

    @property
    def total_chars(self) -> int:
        return self._total_chars

    # ---------------- 内部：切块 ----------------

    def _cut_chunk(self, keep_overlap: bool) -> TextChunk:
        """把缓冲句子打包成一个块（从 emit_cursor 起）。"""
        first = self._pending[self._emit_cursor]
        tokens = 0
        chars = 0
        count = 0
        # 贪心取前缀：加下一句不超过 token/字符预算就继续（单句早已保证 ≤ maxChunkChars）
        while (
            self._emit_cursor + count < len(self._pending)
            and tokens + self._pending[self._emit_cursor + count].tokens
            <= self.opts.max_chunk_tokens
            and chars + js_len(self._pending[self._emit_cursor + count].text)
            <= self.opts.max_chunk_chars
        ):
            sentence = self._pending[self._emit_cursor + count]
            tokens += sentence.tokens
            chars += js_len(sentence.text)
            count += 1
        # 兜底：一句都取不到说明单句本身超预算（超长句已硬切，理论不可达），强制取一句
        if count == 0:
            count = 1
        last = self._pending[self._emit_cursor + count - 1]

        content = "".join(
            s.text for s in self._pending[self._emit_cursor : self._emit_cursor + count]
        )
        chunk_id = str(uuid.uuid4())
        chunk = TextChunk(
            chunk_id=chunk_id,
            doc_id=self.doc_id,
            content=content,
            doc_name=self.doc_name,
            ord=len(self._chunks),  # finish() 时统一最终序号
            heading_path=json.loads(self._current_path_key or "[]"),
            page_start=first.page,
            page_end=last.page,
            char_start=first.start,
            char_end=last.end,
            # JS `tokens || countTokens(content)`：0 视为未填，现场重算
            token_count=tokens or count_tokens(content),
            parent_id=self._group_parent_id or chunk_id,
            prev_id="",
            next_id="",
            content_hash=fnv1a32(content),
        )
        if not self._group_parent_id:
            self._group_parent_id = chunk_id  # 组首块
        self._chunks.append(chunk)
        self._last_chunk = chunk

        # 保留末尾整句作为下一块的开头；但至少前进 1 句（块内句子数 ≤ overlap 时不可全保留）
        keep = min(self.opts.sentence_overlap, count - 1) if keep_overlap else 0
        consumed = count - keep
        for i in range(self._emit_cursor, self._emit_cursor + consumed):
            self._pending_tokens -= self._pending[i].tokens
            self._pending_chars -= js_len(self._pending[i].text)
        self._emit_cursor += consumed
        return chunk

    def _compact_pending(self) -> None:
        """压缩缓冲：已消费前缀过长时删掉、游标归零（start/end 是全文绝对偏移，删除安全）。"""
        if self._emit_cursor < PENDING_COMPACT_THRESHOLD:
            return
        del self._pending[: self._emit_cursor]
        self._emit_cursor = 0
        self.stats.compactions += 1

    def _flush_boundary(self) -> None:
        """结构边界（标题切换 / finish 收尾）：当前缓冲整段切走，不保留重叠。"""
        while self._emit_cursor < len(self._pending):
            self._cut_chunk(False)
        self._pending.clear()
        self._emit_cursor = 0
        self._pending_tokens = 0
        self._pending_chars = 0

    # ---------------- 内部：push ----------------

    def _push_sentence(self, sentence: str, page: int) -> None:
        tokens = count_tokens(sentence)
        self.stats.tokens_counted += 1
        self._pending.append(
            _PendingSentence(
                text=sentence,
                start=self._offset,
                end=self._offset + js_len(sentence),
                page=page,
                tokens=tokens,
            )
        )
        self._offset += js_len(sentence)
        self._pending_tokens += tokens
        self._pending_chars += js_len(sentence)
        self.stats.pending_peak = max(self.stats.pending_peak, len(self._pending))

    def _push_block(self, block: ChunkBlock) -> None:
        text = block.text or ""
        if not text.strip():
            return
        page = block.page if block.page is not None else 1
        path_key = _path_key(block.heading_path)
        self._total_chars += js_len(text)

        # 结构边界：heading 切换前先切走旧组缓冲
        if path_key != self._current_path_key:
            self._flush_boundary()
            self._current_path_key = path_key
            self._group_parent_id = ""
        # 标题块本身不进正文（标题文本在 headingPath 里），只作为结构信号
        if block.type == "heading":
            return

        # 句子切分 + 硬切兜底：单句超过字符硬上限按 maxChunkChars 切开
        for sentence in split_sentences_keep(text):
            piece = sentence
            while js_len(piece) > self.opts.max_chunk_chars:
                self._push_sentence(js_slice(piece, 0, self.opts.max_chunk_chars), page)
                piece = js_slice(piece, self.opts.max_chunk_chars)
            if piece:
                self._push_sentence(piece, page)

        # 打包：预算攒满就切块（token 或字符任一达标）
        while self._emit_cursor < len(self._pending) and (
            self._pending_tokens >= self.opts.max_chunk_tokens
            or self._pending_chars >= self.opts.max_chunk_chars
        ):
            self._cut_chunk(True)

        # 消费完就压缩缓冲，避免纯文本/PDF（headingPath 恒为 []）把全文句子攒到最后
        self._compact_pending()

    def push(self, block: Union[ChunkBlock, str]) -> None:
        """追加一块文本（带页码/标题路径的解析层输出）。"""
        self._push_block(ChunkBlock(text=block) if isinstance(block, str) else block)

    # ---------------- 收尾 ----------------

    def finish(self) -> list[TextChunk]:
        """刷出最后剩余块、补齐序号与前后链接，返回全部分块。"""
        rest = self._pending[self._emit_cursor :]
        rest_text = "".join(s.text for s in rest)
        if rest_text:
            if js_len(rest_text) >= self.opts.min_chunk_chars:
                self._flush_boundary()
            elif self._last_chunk is not None:
                # 尾部残块：与上一块末尾重叠的部分去掉，只补新增部分
                overlap_len = longest_overlap(self._last_chunk.content, rest_text)
                unique = rest_text[overlap_len:]
                if unique:
                    self._last_chunk.content += unique
                    self._last_chunk.char_end = rest[-1].end
                    self._last_chunk.page_end = rest[-1].page
                    self._last_chunk.token_count = count_tokens(
                        self._last_chunk.content
                    )
                    self._last_chunk.content_hash = fnv1a32(self._last_chunk.content)
            else:
                # 全文短于 minChunkChars：单独成块
                self._flush_boundary()

        for i, chunk in enumerate(self._chunks):
            chunk.ord = i
            chunk.prev_id = self._chunks[i - 1].chunk_id if i > 0 else ""
            chunk.next_id = (
                self._chunks[i + 1].chunk_id if i < len(self._chunks) - 1 else ""
            )
        return self._chunks


def create_text_chunks(
    doc_id: str,
    doc_name: str,
    text: str,
    options: ChunkOptions = DEFAULT_CHUNK_OPTIONS,
) -> list[TextChunk]:
    """一次性分块入口（内部复用流式分块器）。"""
    stream = TextChunkStream(doc_id, doc_name, options)
    stream.push(ChunkBlock(text=text, page=1))
    return stream.finish()
