/**
 * 问答 Prompt 构造（纯函数）。
 *
 * 从 llm.worker.ts 抽出来是为了**让答案级评测与线上用同一个 Prompt**：
 * 评测脚本要在 Node 里跑本地 Qwen 生成答案并检查要点命中，
 * 如果它另写一份 Prompt，量出来的就不是线上行为。
 * 之前「两份 tokenize 靠注释同步」「评测抄一份检索实现」都是同一类坑。
 */

/** 喂给模型的编号段落 */
export interface PromptPassage {
  /** 引用编号，从 1 开始 */
  index: number;
  /** 展示标签：文件名 · 章节 · P12 */
  label: string;
  /** 段落正文 */
  text: string;
}

export interface AnswerPromptInput {
  question: string;
  /** 编号段落（与 UI 的引用卡片同源）；为空时走"无文档"分支 */
  passages: PromptPassage[];
  /** 最近对话历史 */
  history?: { role: "user" | "assistant"; content: string }[];
  /** 更早对话的滚动摘要 */
  historySummary?: string;
}

/**
 * 构造 Qwen chat 格式的完整 Prompt。
 * 要求模型先输出 `<think>` 思考再给答案，并逐句标注引用角标。
 */
export function buildAnswerPrompt(input: AnswerPromptInput): string {
  const { question, passages, history, historySummary } = input;

  // 编号段落：每段带 [n] 角标 +「文件名 · 章节 · 页码」标签，
  // 模型据此逐句标注依据，主线程据此收敛校验范围、渲染引用卡片。
  const context = passages.length
    ? passages.map((p) => `[${p.index}] ${p.label}\n${p.text}`).join("\n\n")
    : "";
  const citationRule = passages.length
    ? `引用规则（必须遵守）：
1. 每句话结尾都要标注依据编号，格式 [1]；一句话用了多段依据就写 [1][2]
2. 只能引用【文档内容】中列出的编号，禁止编造编号
3. 文档中没有依据的内容一律不写；整篇都没有依据时只回复：文档中没有找到相关内容
`
    : `文档无相关内容，直接回复：文档中没有找到相关内容，禁止编造信息。
`;
  const historyBlock = (history ?? [])
    .map((turn) => `<|im_start|>${turn.role}\n${turn.content}<|im_end|>`)
    .join("\n");

  return `<|im_start|>system
你是文档问答助手，只能使用【文档内容】回答。
回答前先写出简短思考，再输出答案。
${citationRule}可结合【此前对话摘要】与最近对话理解用户追问中的指代（如"它""上面提到的"），但回答依据仍必须来自文档内容。
【文档内容】
${context}${historySummary ? `\n【此前对话摘要】\n${historySummary}` : ""}
<|im_end|>
${historyBlock ? historyBlock + "\n" : ""}<|im_start|>user
${question}<|im_end|>
<|im_start|>assistant
`;
}
