# 跨语言分块规范（chunker spec）

> 本目录定义「同一份文档，在 TS（浏览器 Lite 模式）与 Python（apps/api 服务端）必须切出**相同分块**」
> 的口径。规范的**执行者**是两端的测试，不是这份文档：
>
> - `apps/web/scripts/parity-dump.ts`：把 TS 端**真实**切块结果导出成 JSON
> - `apps/api/tests/test_chunk_parity.py`：在 Python 端用同一份输入重跑，逐块比对
> - `expected-chunks.json`：fixtures 的期望值冻结副本（无 Node 环境时的兜底；有 Node 时以实时结果为准）
>
> **不一致 = CI 红**（contentHash 不等即失败）。

## 1. 为什么这件事值钱

`contentHash` 是索引的身份证：`packages/eval/embeddings.json` 按 contentHash 存向量，
索引版本按 contentHash 判定复用。两端切块只要差一个字符，向量就对不上、
「同一份语料两端结果可对比」这个前提就没了。

所以本规范不是"代码风格约定"，而是**跨端可比性的前置条件**：先有相同分块，才有相同检索，才有双端对比。

## 2. 参数默认值（唯一来源：`apps/web/src/utils/chunk.ts` 的 `DEFAULT_CHUNK_OPTIONS`）

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `maxChunkTokens` | 256 | 单块 token 预算（`countTokens` 估算口径，bge-small-zh 上限 512 留余量） |
| `minChunkChars` | 100 | 尾部残块低于此值并入上一块 |
| `maxChunkChars` | 600 | 单块字符硬上限（超长单句在此处硬切） |
| `sentenceOverlap` | 2 | 相邻块重叠的**整句**数量 |

索引版本 = `fnv1a32(JSON.stringify([四个参数, EMBEDDING_PREFIX_SCHEME, EMBEDDING_MODEL_NAME]))`。
两者任一变化都会让旧向量不可比，因此该值随分块口径一起被测试钉住（TS `0635b636` / Python 必须相同）。

- `EMBEDDING_MODEL_NAME` = `Xenova/bge-small-zh-v1.5`
- `EMBEDDING_PREFIX_SCHEME` = `passage:none|query:为这个句子生成表示以用于检索相关文章：|verify:none`
  （查询加中文 instruction，文档与校验句不加）

## 3. 切块算法（三级）

1. **结构边界**：解析层产出带 `headingPath` 的块，`headingPath` 变化即 `flushBoundary()` —— 当前缓冲整段切走，**不保留重叠**，且缓冲清空（标题组之间不跨接）。
2. **句子打包**：句子是原子单位，攒到 `maxChunkTokens` 或 `maxChunkChars` 任一达标就切块；块尾保留 `min(sentenceOverlap, 块内句数-1)` 个整句作为下一块开头（至少前进 1 句，防止死循环）。
3. **硬切兜底**：单句长度 > `maxChunkChars` 时按 `maxChunkChars` 逐段硬切，保底不丢内容。

收尾（`finish()`）：

- 剩余缓冲文本长度 ≥ `minChunkChars` → 独立成块；
- 否则若已有块 → 并入**上一块**，并入前先用 `longestOverlap(上一块内容, 残块)` 去掉重复尾巴（重叠整句造成的），并重算 `tokenCount` 与 `contentHash`；
- 全文短于 `minChunkChars` 且无任何块 → 独立成块。

最后统一回填 `ord`（0 起连续）、`prevId`/`nextId`。

### 必须逐字对齐的细节

| 细节 | 规则 |
| --- | --- |
| 句子切分 | `/[^。！？!?；;\n]+[。！？!?；;\n]?|\n+/g`，并丢弃 `trim()` 后为空的片段（分隔符保留在句子里） |
| 空块 | `text.trim() === ""` 的块直接忽略；`type === "heading"` 的块不进正文（标题只在 `headingPath` 里），但**计入 `totalChars`** |
| 页码 | 块首句页码 → `pageStart`，块尾句页码 → `pageEnd`；无分页格式恒为 1；PDF 逐页递增 |
| 偏移 | `charStart`/`charEnd` 是**块首句起点**与**块尾句终点**在清洗后全文中的偏移（清洗 = 解析层做的事，不是分块器做的） |
| 标题路径空洞 | 首个标题是 `###` 时 TS 用 `headingPath.length = level - 1` 造出空洞，经 JSON 往返成为 `null`；Python 必须补 `None`，得到 `[null, null, "标题"]` 而不是 `["标题"]` |
| token 估算 | 见下节；**不是**真实 tokenizer |
| 哈希 | 见下节 |

## 4. `countTokens`（token 估算）

正则估算，不加载真实 tokenizer：

- `/[一-鿿]/g` 每命中一个汉字计 1 token
- `/[a-zA-Z0-9]+(?:\.\d+)?/g` 每命中一段英文/数字串计 1 token
- 其余（标点、空白、其他文字）不计

## 5. `tokenize`（BM25 与词法校验共用）

- 英文/数字：`/[a-z0-9]+(?:\.\d+)?%?/gi`，命中后**转小写**，保留重复（BM25 的 TF 需要重复）
- 中文连续段：`/[一-鿿]+/g`，每段产出**二元组**（bigram）：`中文段` → `中文`、`文段`；长度为 1 的段保留该单字
- `tokenizeSet` = `tokenize` 去重后的集合

## 6. `fnv1a32`（contentHash / 索引版本 / 去重哈希）

```
hash = 0x811c9dc5
对每个 **UTF-16 码元**（不是码点！）:
    hash ^= 码元
    hash = Math.imul(hash, 0x01000193)   // 32 位有符号乘法，等价于 (hash * 0x01000193) & 0xFFFFFFFF
返回 (hash >>> 0).toString(16).padStart(8, "0")
```

⚠ **Python 必须按 UTF-16 编码单元迭代**（`text.encode("utf-16-le")` 后按 2 字节一组取无符号短整型），
不能用码点：非 BMP 字符（emoji、𝄞 等）在 UTF-16 里是代理对（2 个码元），按码点哈希结果不同。
`fixtures/nonbmp.md` 专门用来抓这个差异。

## 7. 两条切块路径（都测）

| 路径 | 输入 | 用途 |
| --- | --- | --- |
| **评测路径** | `normalizeMarkdownForEval(raw)`（逐行删掉 `^#{1,6}\s` 标题行、折叠 3+ 连续空行、trim）→ 整篇作为**一个块**（`headingPath: []`）→ 分块 | `packages/eval/embeddings.json` 的 119 个 contentHash 就是这个路径产生的 |
| **摄取路径** | `markdownToBlocks(raw)`（标题建栈 → heading 块 + 段落块，各带 `headingPath`）→ 逐块 push → 分块 | 真实上传文档的路径；标题边界会改变分组与切块 |

解析层规则（摄取路径，`apps/web/src/utils/parseBlocks.ts`）：

- 先删 HTML 注释 `<!--[\s\S]*?-->`，再剥 YAML front matter（仅当 `^---\r?\n([\s\S]*?)\r?\n---` 匹配上、且中间内容为空或含 `^[\w-]+\s*:` 才剥离），再删行尾空白
- 标题行判定：`^(#{1,6})\s+(.*)$`（必须在行首；`#无空格`、7 个 `#` 都不是标题），标题文本 `trim()`
- 标题处理顺序：**先 flush 当前段落**，再 `headingPath.length = 级数 - 1`（造空洞见上表），再 push 标题，再产出 heading 块
- 正文：连续非空行（各自 `trim()`）以 `\n` 连接成一个段落块；空行 flush
- 段落块文本 = `段落行.join("\n").trim()`，为空则不产出

## 8. 明确不承诺的部分

- **PDF / DOCX 解析不一致**：浏览器用 pdfjs / mammoth，服务端用 Python 库，同一份 PDF 的文本抽取结果**不保证相同**（换行、连字符、阅读顺序都可能不同）。因此跨端一致性只承诺 **Markdown / TXT 的解析 + 两端的全部切块逻辑**；PDF/DOCX 的切块一致性不纳入门槛。
- **chunkId 不确定**：TS 用 `crypto.randomUUID()`，两端不比对 id；引用锚点用各端自己的 id。
- **上传路径的编码判定**：浏览器按 BOM / 默认解码读文本文件，Node 与 Python 默认 UTF-8；非 UTF-8 文本文件的抽取结果可能不同。

## 9. 怎么跑

```bash
# TS 端导出（真实实现，不复制逻辑）
cd apps/web
npm run parity:dump -- --section chunks --mode ingest --corpus ../../packages/chunker-spec/fixtures
npm run parity:dump -- --section chunks --mode eval   --corpus ../../packages/eval/corpus

# Python 端比对（含语料 + fixtures、两条路径）
cd apps/api
uv run pytest tests/test_chunk_parity.py -v
```

需要重建 `expected-chunks.json`（改了 fixtures 或分块器）时，用第 9 节第一条命令导出后按 `modes.ingest / modes.eval` 结构重写；
`apps/api/tests/test_chunk_parity.py` 会同时校验冻结副本与实时结果，两者不一致同样报错。
