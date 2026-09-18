import { pipeline, env, TextStreamer } from "@huggingface/transformers";

//模型从本地 /models 加载,禁止远程，避免走 HF 官网
env.localModelPath = "/models";
env.allowLocalModels = true;
env.allowRemoteModels = false;

//核心设计：惰性加载 + Promise 缓存（单例模式）
let generatorPromise: Promise<any> | null = null;
//存的是 Promise，不是模型实例本身。好处：就算多个请求同时调用 `getGenerator`，也只会触发一次 pipeline 加载。
async function getGenerator(
  onProgress: (progress: number) => void,
): Promise<any> {
  if (!generatorPromise) {
    // 加载前检查内存（WASM 模型可能需数百 MB，提前预判）
    if (!checkMemoryBudget()) {
      throw new Error(
        "浏览器内存不足，无法加载模型。请关闭其他标签页后重试。",
      );
    }

    generatorPromise = pipeline(
      "text-generation",
      "onnx-community/Qwen2.5-0.5B-Instruct",
      {
        dtype: "q4",
        progress_callback: (info: any) => {
          if (info.status === "progress" && typeof info.progress === "number") {
            onProgress(info.progress);
          }
        },
      } as any,
    );
    // 加载失败清空缓存，允许下次重试，避免永久卡死在 rejected Promise 上
    generatorPromise.catch(() => {
      generatorPromise = null;
    });
  }
  return generatorPromise;
}

/**
 * 检查浏览器是否有足够内存加载模型。
 * performance.memory 是 Chrome 非标准 API，有则用，无则跳过（不阻塞）。
 */
function checkMemoryBudget(): boolean {
  const mem = (performance as any).memory;
  if (!mem) return true; // 无法检测，放行
  // jsHeapSizeLimit：浏览器分配给 JS 堆的上限；usedJSHeapSize：当前已用
  // 注意：WASM/ArrayBuffer 往往不计入 usedJSHeapSize，此检查只能粗筛，不能保证加载成功
  const available = mem.jsHeapSizeLimit - mem.usedJSHeapSize;
  // Qwen 0.5B q4 约需 300~400MB，阈值设 450MB 给余量
  return available > 450 * 1024 * 1024;
}

self.onmessage = async (e: MessageEvent) => {
  //`self` 在 Web Worker 里面代表 worker 全局对象，相当于主线程的 `window`。
  const data = e.data as {
    question?: string;
    contextChunks?: string[];
    history?: { role: "user" | "assistant"; content: string }[];
    historySummary?: string;
    text?: string;
    type?: string;
  };

  // 历史摘要压缩模式：非流式、低温、短输出，产出滚动摘要
  if (data.type === "summarize" && typeof data.text === "string") {
    const id = data.id;
    try {
      self.postMessage({ type: "loading", id });
      const generator = await getGenerator(() => {});
      const summaryPrompt = `<|im_start|>system
你是对话压缩助手。把用户提供的历史对话（可能包含一段已有摘要和新增对话）压缩成一段连贯的中文摘要，只保留：用户的关键问题、已确认的结论、重要数字与专有名词、尚未解决的问题。要求：不要新增信息、不要分点罗列、不要寒暄，直接输出摘要正文，300字以内。<|im_end|>
<|im_start|>user
${data.text}<|im_end|>
<|im_start|>assistant
`;
      const output = await generator(summaryPrompt, {
        max_new_tokens: 320,
        do_sample: false,
        return_full_text: false,
      });
      self.postMessage({
        type: "done",
        id,
        text: String(output[0].generated_text ?? "").trim(),
      });
    } catch (err: any) {
      self.postMessage({ type: "error", id, error: err?.message || String(err) });
    }
    return;
  }

  // 子问题拆分模式：非流式、低温，输出 JSON 数组
  if (data.type === "split" && typeof data.text === "string") {
    const id = data.id;
    try {
      self.postMessage({ type: "loading", id });
      const generator = await getGenerator(() => {});
      const splitPrompt = `<|im_start|>system
你是一个问题拆分助手。把用户的问题拆分成最多 3 个独立的子问题，用于后续检索。

规则：
1. 最多 3 个子问题，能不拆就不拆
2. 每个子问题必须是完整、独立可检索的短句
3. 只输出 JSON 数组，不要解释、不要寒暄

示例：
问："对比文档中 A 产品和 B 产品的性能差异"
输出：["A 产品的性能指标","B 产品的性能指标","A 产品与 B 产品的性能对比"]<|im_end|>
<|im_start|>user
${data.text}<|im_end|>
<|im_start|>assistant
`;
      const output = await generator(splitPrompt, {
        max_new_tokens: 256,
        do_sample: false,
        return_full_text: false,
      });
      self.postMessage({
        type: "done",
        id,
        text: String(output[0].generated_text ?? "").trim(),
      });
    } catch (err: any) {
      self.postMessage({ type: "error", id, error: err?.message || String(err) });
    }
    return;
  }

  // 信息充足性判断模式：非流式、低温，只输出「足够」或「不足」
  if (
    data.type === "evaluate" &&
    typeof data.text === "string" &&
    Array.isArray(data.contextChunks)
  ) {
    const id = data.id;
    try {
      self.postMessage({ type: "loading", id });
      const generator = await getGenerator(() => {});
      const context = (data.contextChunks as string[]).join("\n");
      const evalPrompt = `<|im_start|>system
你是信息充足性判断助手。根据【检索到的文档片段】，判断是否能完整回答用户的问题。

规则：
1. 只能输出「足够」或「不足」两个词之一
2. 片段包含回答问题所需的关键信息 → 输出「足够」
3. 片段缺失关键信息、只有部分内容、或与问题不相关 → 输出「不足」
4. 禁止输出其他任何文字<|im_end|>
<|im_start|>user
【用户问题】
${data.text}
【检索到的文档片段】
${context}<|im_end|>
<|im_start|>assistant
`;
      const output = await generator(evalPrompt, {
        max_new_tokens: 32,
        do_sample: false,
        return_full_text: false,
      });
      self.postMessage({
        type: "done",
        id,
        text: String(output[0].generated_text ?? "").trim(),
      });
    } catch (err: any) {
      self.postMessage({ type: "error", id, error: err?.message || String(err) });
    }
    return;
  }

  // 手动卸载模型（当前架构 LLM worker 是一次性的，done 时已 terminate；
  // 此消息以备未来改为常驻 worker 时使用）
  if (data.type === "unload-model") {
    const oldPromise = generatorPromise;
    generatorPromise = null;
    if (oldPromise) {
      try {
        const gen = await oldPromise;
        if (gen && typeof gen.dispose === "function") {
          await gen.dispose();
        } else if (gen?.model && typeof gen.model.dispose === "function") {
          await gen.model.dispose();
        }
      } catch {
        // 忽略
      }
    }
    self.postMessage({ type: "unloaded" });
    return;
  }

  const { question, contextChunks, history, historySummary } = data as {
    question: string;
    contextChunks: string[];
    history?: { role: "user" | "assistant"; content: string }[];
    historySummary?: string;
  };
  const id = data.id;

  try {
    // 通知主线程：开始加载模型
    self.postMessage({ type: "loading", id });

    const generator = await getGenerator((progress) => {
      self.postMessage({ type: "load-progress", progress, id });
    }); //模型加载完成之后，generator 就是 transformers 的 pipeline 实例，用来做文本生成。

    const context = contextChunks.join("\n"); //把检索出来的多个文档切片，用换行拼接成一整段文本，放进 prompt 作为参考文档。
    // 最近对话历史（旧对话已在主线程压缩为 historySummary），支持追问中的指代理解
    const historyBlock = (history ?? [])
      .map((turn) => `<|im_start|>${turn.role}\n${turn.content}<|im_end|>`)
      .join("\n");
    // 让模型先用 <think>...</think> 输出思考过程，再给最终回答
    const prompt = `<|im_start|>system
你是文档问答助手，只能使用【文档内容】回答。
回答前先在写出简短思考，再输出答案。
可结合【此前对话摘要】与最近对话理解用户追问中的指代（如"它""上面提到的"），但回答依据仍必须来自文档内容。
文档无相关内容，直接回复：文档中没有找到相关内容，禁止编造信息。
【文档内容】
${context}${
      historySummary
        ? `\n【此前对话摘要】\n${historySummary}`
        : ""
    }
<|im_end|>
${historyBlock ? historyBlock + "\n" : ""}<|im_start|>user
${question}<|im_end|>
<|im_start|>assistant
`;

    // 通知：模型已加载完成，准备生成
    self.postMessage({ type: "generating", id });

    const streamer = new TextStreamer(generator.tokenizer, {
      //`TextStreamer`：transformers.js 的流式输出工具。
      skip_prompt: true, //`skip_prompt:true`：不要把我们输入的 system+user 提示词原样吐出来，只输出模型新生成的内容
      skip_special_tokens: true, //`skip_special_tokens:true`：过滤掉 `<|im_start|>`、`<|im_end|>` 这类模型特殊标记，不展示给用户
      callback_function: (text: string) => {
        // 直接把原始 token 转发给主线程，由主线程解析 think 标签
        self.postMessage({ type: "token", text, id });
      },
    });

    const output = await generator(prompt, {
      max_new_tokens: 768, //限制模型最多新生成 768 个 token，防止无限长输出
      temperature: 0.4, //`temperature:0.4`：越低，回答越确定、越严谨；越高创造性越强。RAG 文档问答一般用 0.2~0.5。
      do_sample: true, //`do_sample:false`：关闭随机采样，模型每次相同输入输出更稳定，适合问答场景。
      streamer, //绑定上面的流式回调，一边生成一边实时推送 token。
      return_full_text: false, //返回结果只包含模型新生成的文字，不包含输入的 prompt。
    });

    const finalText: string = output[0].generated_text;
    self.postMessage({ type: "done", id, text: finalText });
  } catch (err: any) {
    self.postMessage({ type: "error", id, error: err?.message || String(err) });
  }
};

// 兜底：Worker 内未捕获的 Promise rejection（第三方库内部抛出但没 catch）
self.addEventListener("unhandledrejection", (event) => {
  const msg = event.reason instanceof Error
    ? event.reason.message
    : String(event.reason);
  self.postMessage({
    type: "error",
    error: `模型线程发生未捕获异常：${msg}`,
  });
  event.preventDefault();
});
