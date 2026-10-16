import { pipeline, env, TextStreamer } from "@huggingface/transformers";
import type {
  PretrainedModelOptions,
  ProgressInfo,
  TextGenerationPipeline,
} from "@huggingface/transformers";
import { buildAnswerPrompt } from "../utils/answerPrompt";
import { buildExpandPrompt } from "../utils/expandPrompt";
import type { DeviceKind } from "../utils/deviceAccel";

//模型从本地 /models 加载,禁止远程，避免走 HF 官网
//用 BASE_URL 拼相对路径：部署到子路径（如 /app/）时不会 404
env.localModelPath = `${import.meta.env.BASE_URL}models`;
env.allowLocalModels = true;
env.allowRemoteModels = false;

/** Chrome 非标准 performance.memory 字段（其他浏览器无此字段，访问前先探测） */
interface ChromeMemoryInfo {
  jsHeapSizeLimit: number;
  usedJSHeapSize: number;
}

/* ================= 请求协议（判别联合：每种任务只带自己需要的字段） ================= */

/** 对话历史轮次（追问改写用） */
interface LlmHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

/** 生成答案的编号段落：[1] 文件名 · 章节 · P12 + 正文 */
interface LlmPassage {
  index: number;
  text: string;
  label: string;
}

export type LlmRequest =
  /** 历史对话压缩为滚动摘要（非流式） */
  | { type: "summarize"; id?: number; text: string }
  /** 子问题拆分（非流式，输出 JSON 数组） */
  | { type: "split"; id?: number; text: string }
  /** 多轮追问改写为独立问题（非流式） */
  | { type: "condense"; id?: number; text: string; history?: LlmHistoryTurn[] }
  /** 查询扩展：把口语化提问改写成术语化检索词（返回多行，逐行一条） */
  | { type: "expand"; id?: number; text: string; max?: number }
  /** 信息充足性判断（非流式，只输出「足够/不足」） */
  | { type: "evaluate"; id?: number; text: string; contextChunks: string[] }
  /** 释放模型 */
  | { type: "unload-model"; id?: number }
  /** 切换推理设备（WebGPU / WASM）：设备真变了才释放模型，下次请求重新加载 */
  | { type: "set-device"; device: DeviceKind }
  /** 流式生成答案（唯一会推 token 的任务） */
  | {
      type: "generate";
      id?: number;
      question: string;
      /** 编号段落；缺省时退回旧的纯文本分片通道 */
      passages?: LlmPassage[];
      contextChunks?: string[];
      history?: LlmHistoryTurn[];
      historySummary?: string;
    };

/** worker -> 主线程响应 */
export type LlmResponse =
  | { type: "loading"; id?: number }
  | { type: "load-progress"; id?: number; progress: number }
  /** 设备回报（无 id 广播）：模型真正加载 + 预热成功后通告实际生效设备 */
  | {
      type: "device";
      channel: "llm";
      device: DeviceKind;
      fallbackReason: string | null;
    }
  | { type: "generating"; id?: number }
  | { type: "token"; id?: number; text: string }
  | { type: "done"; id?: number; text: string }
  | { type: "unloaded"; id?: number }
  | { type: "error"; id?: number; error: string };

//核心设计：惰性加载 + Promise 缓存（单例模式）
let generatorPromise: Promise<TextGenerationPipeline> | null = null;
/** 期望设备（主线程 set-device 下发；实际生效设备以加载/预热结果为准） */
let preferredDevice: DeviceKind = "wasm";
/** WebGPU 不可用原因：非 null 时不再尝试 GPU，仅 set-device 复位 */
let webgpuBlockedReason: string | null = null;
/**
 * 切换设备时被摘下的旧模型：**延后到下一次加载时才释放**。
 * set-device 消息可能在一次流式生成中途到达，立即 dispose 会打断正在跑的推理；
 * 推迟到下次加载时，推理锁保证此刻没有在跑的推理。
 */
let staleGenerators: Promise<TextGenerationPipeline>[] = [];

/** 释放所有被摘下的旧模型（在新模型加载前调用） */
async function releaseStaleGenerators(): Promise<void> {
  const olds = staleGenerators;
  staleGenerators = [];
  for (const oldPromise of olds) {
    try {
      const gen = await oldPromise;
      if (gen && typeof gen.dispose === "function") await gen.dispose();
    } catch {
      // 旧实例加载中失败，忽略
    }
  }
}

/** 回报「本通道当前生效设备」：埋点按设备分桶，面板显示回退原因 */
function reportDevice(device: DeviceKind, fallbackReason: string | null): void {
  self.postMessage({ type: "device", channel: "llm", device, fallbackReason });
}

/**
 * 按指定设备创建文本生成 pipeline。
 * webgpu 需要显式 dtype：v4 里 wasm 默认 q8、webgpu 默认 fp32，
 * 本地只托管了 model_q4.onnx，不给 dtype 会去找不存在的 model.onnx。
 */
function createGenerator(
  device: DeviceKind,
  onProgress: (progress: number) => void,
): Promise<TextGenerationPipeline> {
  const options: Record<string, unknown> = {
    dtype: "q4",
    progress_callback: (info: ProgressInfo) => {
      if (info.status === "progress" && typeof info.progress === "number") {
        onProgress(info.progress);
      }
    },
  };
  if (device === "webgpu") options.device = "webgpu";
  console.log(`开始加载本地模型 Qwen2.5-0.5B-Instruct（${device}）`);
  return pipeline(
    "text-generation",
    "onnx-community/Qwen2.5-0.5B-Instruct",
    options as unknown as PretrainedModelOptions,
  );
}

/** 预热：跑一次 1 token 生成，暴露「能建会话但跑不动」的着色器/算子问题 */
async function warmupGenerator(generator: TextGenerationPipeline): Promise<void> {
  await generator("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", {
    max_new_tokens: 1,
    do_sample: false,
    return_full_text: false,
  });
}

/**
 * 按偏好设备加载 LLM：WebGPU 失败（不支持/显存不足/着色器编译失败）自动回退 WASM，
 * 并把原因回报主线程 —— 回退可见，不静默降级。
 * 两种设备用的是同一个 q4 权重文件，回退不改变模型能力。
 */
async function loadGenerator(
  onProgress: (progress: number) => void,
): Promise<TextGenerationPipeline> {
  // 换设备时摘下的旧实例在这里释放（此刻没有推理在跑）
  await releaseStaleGenerators();
  if (preferredDevice === "webgpu" && !webgpuBlockedReason) {
    let attempt: TextGenerationPipeline | null = null;
    try {
      attempt = await createGenerator("webgpu", onProgress);
      await warmupGenerator(attempt);
      reportDevice("webgpu", null);
      console.log("[device] 本地 LLM 运行在 WebGPU（dtype=q4）");
      return attempt;
    } catch (err) {
      webgpuBlockedReason = err instanceof Error ? err.message : String(err);
      console.warn("WebGPU 加载/预热失败，回退 WASM 生成：", webgpuBlockedReason);
      // 失败实例必须释放：WebGPU 侧可能已占住显存/session，留着会拖垮随后的 WASM 加载
      try {
        if (attempt && typeof attempt.dispose === "function") {
          await attempt.dispose();
        }
      } catch {
        // 释放失败忽略
      }
    }
  }
  const generator = await createGenerator("wasm", onProgress);
  reportDevice("wasm", preferredDevice === "webgpu" ? webgpuBlockedReason : null);
  return generator;
}

async function getGenerator(
  onProgress: (progress: number) => void,
): Promise<TextGenerationPipeline> {
  if (!generatorPromise) {
    // 加载前检查内存（WASM 模型可能需数百 MB，提前预判）
    if (!checkMemoryBudget()) {
      throw new Error(
        "浏览器内存不足，无法加载模型。请关闭其他标签页后重试。",
      );
    }

    generatorPromise = loadGenerator(onProgress);
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
  const perf = performance as unknown as { memory?: ChromeMemoryInfo };
  const mem = perf.memory;
  if (!mem) return true; // 无法检测，放行
  // jsHeapSizeLimit：浏览器分配给 JS 堆的上限；usedJSHeapSize：当前已用
  // 注意：WASM/ArrayBuffer 往往不计入 usedJSHeapSize，此检查只能粗筛，不能保证加载成功
  const available = mem.jsHeapSizeLimit - mem.usedJSHeapSize;
  // Qwen 0.5B q4 约需 300~400MB，阈值设 450MB 给余量
  return available > 450 * 1024 * 1024;
}

self.onmessage = async (e: MessageEvent) => {
  //`self` 在 Web Worker 里面代表 worker 全局对象，相当于主线程的 `window`。
  // 请求 id 由主线程 agentPlanner 路由响应；协议见 LlmRequest / LlmResponse
  // 判别联合：data.type 一出，TS 自动收窄到该分支的字段
  const data: LlmRequest = e.data as LlmRequest;

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
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: "error", id, error: message });
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
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: "error", id, error: message });
    }
    return;
  }

  // 查询扩展模式（口语化 → 术语化）：输出多行改写，逐行一条
  if (data.type === "expand" && typeof data.text === "string") {
    const id = data.id;
    try {
      self.postMessage({ type: "loading", id });
      const generator = await getGenerator(() => {});
      const prompt = buildExpandPrompt({ question: data.text, max: data.max });
      const output = await generator(prompt, {
        // 贪心：查询扩展要**可复现**（评测里固化的改写必须与线上一致）
        max_new_tokens: 128,
        do_sample: false,
        return_full_text: false,
      });
      self.postMessage({
        type: "done",
        id,
        text: String(output[0].generated_text ?? "").trim(),
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: "error", id, error: message });
    }
    return;
  }

  // 查询改写模式（多轮追问）：把含代词的追问 condense 成独立可检索的问题
  if (data.type === "condense" && typeof data.text === "string") {
    const id = data.id;
    try {
      self.postMessage({ type: "loading", id });
      const generator = await getGenerator(() => {});
      const turns = Array.isArray(data.history) ? data.history : [];
      const historyBlock = turns
        .map((t) => `${t.role === "user" ? "用户" : "助手"}：${t.content}`)
        .join("\n");
      const condensePrompt = `<|im_start|>system
你是查询改写助手。请结合【最近对话】，把用户的【当前问题】改写成一个独立、完整、可直接检索的问题。

规则：
1. 把代词（它、这个、上面提到的、该方法等）替换成对话中明确指代的对象
2. 补齐省略的主语与限定词，使问题脱离上下文也能被理解
3. 只输出改写后的问题本身，不要解释、不要引号、不要任何前缀
4. 当前问题本身已独立完整时，原样输出
5. 不超过 60 字<|im_end|>
<|im_start|>user
【最近对话】
${historyBlock || "（无）"}
【当前问题】
${data.text}<|im_end|>
<|im_start|>assistant
`;
      const output = await generator(condensePrompt, {
        max_new_tokens: 96,
        do_sample: false,
        return_full_text: false,
      });
      self.postMessage({
        type: "done",
        id,
        text: String(output[0].generated_text ?? "").trim(),
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: "error", id, error: message });
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
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: "error", id, error: message });
    }
    return;
  }

  // 切换推理设备：摘掉当前模型（延后释放），下次请求按新设备重新加载
  if (data.type === "set-device") {
    const device = data.device;
    if (device !== preferredDevice) {
      preferredDevice = device;
      // 重新开关 = 再给 WebGPU 一次机会（上次可能只是显存被别的标签页占着）
      webgpuBlockedReason = null;
      const oldPromise = generatorPromise;
      generatorPromise = null;
      if (oldPromise) staleGenerators.push(oldPromise);
      console.log(
        `[device] 本地 LLM 设备偏好切换为 ${device}，下次请求时重新加载`,
      );
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
    // 换设备时摘下的旧模型一并释放，避免残留占用显存/堆
    await releaseStaleGenerators();
    self.postMessage({ type: "unloaded" });
    return;
  }

  // 显式声明 generate 分支的请求类型，收窄后字段都是确定的（不再是可选属性）
  const generateReq: Extract<LlmRequest, { type: "generate" }> = data as Extract<
    LlmRequest,
    { type: "generate" }
  >;
  const { question, passages, contextChunks, history, historySummary } =
    generateReq;
  const id = generateReq.id;

  try {
    // 通知主线程：开始加载模型
    self.postMessage({ type: "loading", id });

    const generator = await getGenerator((progress) => {
      self.postMessage({ type: "load-progress", progress, id });
    }); //模型加载完成之后，generator 就是 transformers 的 pipeline 实例，用来做文本生成。

    // Prompt 构造抽到 src/utils/answerPrompt.ts（纯函数）：
    // 答案级评测脚本要在 Node 里复现同一条 Prompt 才能量到线上行为
    const numbered = passages ?? [];
    const prompt = buildAnswerPrompt({
      question,
      passages: numbered.length
        ? numbered
        : // 旧通道：没有编号段落时退回纯文本分片
          (contextChunks ?? []).map((text, i) => ({
            index: i + 1,
            label: "",
            text,
          })),
      history,
      historySummary,
    });

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
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: "error", id, error: message });
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
