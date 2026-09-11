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
    generatorPromise = pipeline(
      //`pipeline` 来自 `@xenova/transformers`，用来创建推理流水线：
      "text-generation", //第一个参数 `"text-generation"`：任务类型，文本生成（大模型对话）
      "onnx-community/Qwen2.5-0.5B-Instruct", //第二个参数：模型 ID，对应你放在 `public/models` 下的本地模型文件夹名称
      {
        dtype: "q4", //加载**4bit 量化版本的 onnx 权重文件 `model_q4.onnx`**，体积更小，适合 WASM CPU 运行
        progress_callback: (info: any) => {
          if (info.status === "progress" && typeof info.progress === "number") {
            onProgress(info.progress);
          }
        },
      } as any,
    );
  }
  return generatorPromise; //pipeline 一旦开始执行，就算还没加载完成，generatorPromise 也不再是 null。并发调用 getGenerator 都会复用这同一个 Promise，不会重复加载模型。
}

self.onmessage = async (e: MessageEvent) => {
  //`self` 在 Web Worker 里面代表 worker 全局对象，相当于主线程的 `window`。
  const { question, contextChunks } = e.data as {
    question: string;
    contextChunks: string[];
  };

  try {
    // 通知主线程：开始加载模型
    self.postMessage({ type: "loading" });

    const generator = await getGenerator((progress) => {
      self.postMessage({ type: "load-progress", progress });
    }); //模型加载完成之后，generator 就是 transformers 的 pipeline 实例，用来做文本生成。

    const context = contextChunks.join("\n"); //把检索出来的多个文档切片，用换行拼接成一整段文本，放进 prompt 作为参考文档。
    // 让模型先用 <think>...</think> 输出思考过程，再给最终回答
    const prompt = `<|im_start|>system
你是文档问答助手，只能使用【文档内容】回答。
回答前先在写出简短思考，再输出答案。
文档无相关内容，直接回复：文档中没有找到相关内容，禁止编造信息。
【文档内容】
${context}
<|im_end|>
<|im_start|>user
${question}<|im_end|>
<|im_start|>assistant
`;

    // 通知：模型已加载完成，准备生成
    self.postMessage({ type: "generating" });

    const streamer = new TextStreamer(generator.tokenizer, {
      //`TextStreamer`：transformers.js 的流式输出工具。
      skip_prompt: true, //`skip_prompt:true`：不要把我们输入的 system+user 提示词原样吐出来，只输出模型新生成的内容
      skip_special_tokens: true, //`skip_special_tokens:true`：过滤掉 `<|im_start|>`、`<|im_end|>` 这类模型特殊标记，不展示给用户
      callback_function: (text: string) => {
        // 直接把原始 token 转发给主线程，由主线程解析 think 标签
        console.log("模型流式原始token：", JSON.stringify(text));
        self.postMessage({ type: "token", text });
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
    self.postMessage({ type: "done", text: finalText });
  } catch (err: any) {
    self.postMessage({ type: "error", error: err?.message || String(err) });
  }
};
