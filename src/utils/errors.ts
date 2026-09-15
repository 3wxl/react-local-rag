/**
 * 分类错误体系：统一错误码 + 友好提示，UI 层只需按 category 决定展示样式。
 *
 * 设计原则：
 * - Worker 崩溃、模型加载失败、IndexedDB 异常、PDF 损坏、备份损坏 → 各自 category
 * - 每个 AppError 带 userMessage（可直接展示给用户）和 hint（可选恢复建议）
 * - 面试常问"内存不足模型加载崩了怎么处理"→ category="model-load" + hint 提示关标签页/换小模型
 */

export type ErrorCategory =
  | "model-load" // 模型加载失败（文件缺失、内存不足、不支持的设备）
  | "model-inference" // 模型推理失败（生成中断、输出异常）
  | "embedding" // 向量化/检索失败
  | "worker-crash" // Worker 线程崩溃（unhandled error）
  | "worker-timeout" // 主线程等待 Worker 响应超时
  | "indexeddb" // IndexedDB 打开/读写失败
  | "storage-full" // 存储空间已满
  | "pdf-parse" // PDF 解析失败（文件损坏、格式不支持）
  | "file-too-large" // 文件过大，内存溢出风险
  | "backup" // 备份导入损坏/版本不兼容
  | "unknown"; // 兜底

export class AppError extends Error {
  readonly category: ErrorCategory;
  readonly userMessage: string;
  readonly hint?: string;

  constructor(
    category: ErrorCategory,
    userMessage: string,
    options?: { cause?: unknown; hint?: string },
  ) {
    super(userMessage, options);
    this.name = "AppError";
    this.category = category;
    this.userMessage = userMessage;
    this.hint = options?.hint;
  }
}

/* ---------- 工厂方法：常见错误场景一键构造 ---------- */

/** 模型加载失败：识别内存不足/文件缺失并给针对性提示 */
export function modelLoadError(err: unknown): AppError {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();

  if (lower.includes("memory") || lower.includes("oom") || lower.includes("out of memory")) {
    return new AppError(
      "model-load",
      "模型加载失败：浏览器内存不足",
      {
        cause: err,
        hint: "请关闭其他标签页后刷新重试，或换用更小的模型权重文件。",
      },
    );
  }
  if (lower.includes("fetch") || lower.includes("not found") || lower.includes("404") || lower.includes("network")) {
    return new AppError(
      "model-load",
      "模型加载失败：未找到模型文件",
      {
        cause: err,
        hint: "请确认 public/models 目录下已放置对应模型，参考 README 模型加载说明。",
      },
    );
  }
  if (lower.includes("webgpu") || lower.includes("gpu")) {
    return new AppError(
      "model-load",
      "模型加载失败：当前环境不支持所需的计算后端",
      {
        cause: err,
        hint: "请使用支持 WebGPU 的浏览器，或切换到 WASM 后端。",
      },
    );
  }
  return new AppError("model-load", `模型加载失败：${msg}`, { cause: err });
}

/** 模型推理失败 */
export function modelInferenceError(err: unknown): AppError {
  const msg = err instanceof Error ? err.message : String(err);
  return new AppError("model-inference", `模型推理失败：${msg}`, { cause: err });
}

/** 向量化/检索失败 */
export function embeddingError(err: unknown): AppError {
  const msg = err instanceof Error ? err.message : String(err);
  return new AppError("embedding", `向量化或检索失败：${msg}`, { cause: err });
}

/** Worker 线程崩溃 */
export function workerCrashError(workerName: string, err: unknown): AppError {
  const msg = err instanceof Error ? err.message : String(err);
  return new AppError(
    "worker-crash",
    `${workerName}线程崩溃：${msg}`,
    {
      cause: err,
      hint: "请刷新页面重试。如果反复出现，可能是浏览器内存不足。",
    },
  );
}

/** Worker 响应超时 */
export function workerTimeoutError(workerName: string, timeoutMs: number): AppError {
  return new AppError(
    "worker-timeout",
    `${workerName}线程响应超时（${Math.round(timeoutMs / 1000)}秒无响应）`,
    {
      hint: "可能是模型推理耗时过长或线程卡死，请刷新页面后重试。",
    },
  );
}

/** IndexedDB 异常：识别存储空间满 */
export function indexedDBError(err: unknown): AppError {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  if (lower.includes("quota") || lower.includes("storage") || lower.includes("full")) {
    return new AppError(
      "storage-full",
      "浏览器存储空间已满，无法保存数据",
      {
        cause: err,
        hint: "请在浏览器设置中清理站点数据，或删除不需要的会话后重试。",
      },
    );
  }
  return new AppError("indexeddb", `数据存储失败：${msg}`, { cause: err });
}

/** PDF 解析失败 */
export function pdfParseError(err: unknown): AppError {
  const msg = err instanceof Error ? err.message : String(err);
  return new AppError(
    "pdf-parse",
    `文档解析失败：${msg}`,
    {
      cause: err,
      hint: "请确认文件未损坏且为有效的 PDF/TXT 格式。",
    },
  );
}

/** 文件过大 */
export function fileTooLargeError(sizeMB: number, limitMB: number): AppError {
  return new AppError(
    "file-too-large",
    `文件过大（${sizeMB.toFixed(1)}MB），超过建议上限 ${limitMB}MB`,
    {
      hint: "大文件可能导致浏览器内存溢出，请拆分后再上传。",
    },
  );
}
