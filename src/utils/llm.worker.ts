import { pipeline, env, TextStreamer } from "@huggingface/transformers";

env.localModelPath = "/models";
env.allowLocalModels = true;
env.allowRemoteModels = false;

let generatorPromise: Promise<any> | null = null;

async function getGenerator(
  onProgress: (progress: number) => void,
): Promise<any> {
  if (!generatorPromise) {
    generatorPromise = pipeline(
      "text-generation",
      "onnx-community/Qwen2.5-0.5B-Instruct",
      {
        quantized: true,
        progress_callback: (info: any) => {
          if (info.status === "progress" && typeof info.progress === "number") {
            onProgress(info.progress);
          }
        },
      } as any,
    );
  }
  return generatorPromise;
}

self.onmessage = async (e: MessageEvent) => {
  const { question, contextChunks } = e.data as {
    question: string;
    contextChunks: string[];
  };

  try {
    self.postMessage({ type: "loading" });

    const generator = await getGenerator((progress) => {
      self.postMessage({ type: "load-progress", progress });
    });

    const context = contextChunks.join("\n");
    const prompt = `<|im_start|>system
你是文档问答助手，请严格依据下面【文档内容】回答用户问题。
如果文档中没有相关信息，直接回复：文档中没有找到相关内容，不要编造任何信息。
【文档内容】
${context}
<|im_end|>
<|im_start|>user
${question}<|im_end|>
<|im_start|>assistant
`;

    self.postMessage({ type: "generating" });

    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        self.postMessage({ type: "token", text });
      },
    });

    const output = await generator(prompt, {
      max_new_tokens: 512,
      temperature: 0.3,
      do_sample: false,
      streamer,
      return_full_text: false,
    });

    const finalText: string = output[0].generated_text;
    self.postMessage({ type: "done", text: finalText });
  } catch (err: any) {
    self.postMessage({ type: "error", error: err?.message || String(err) });
  }
};
