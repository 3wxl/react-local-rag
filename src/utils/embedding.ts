import { pipeline, env } from "@huggingface/transformers";

// 配置：读取public下本地模型，关闭远程拉取，不再需要hf代理
env.localModelPath = "/models";
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

let embedderPromise: Promise<any> | null = null;
async function getEmbedder() {
  if (!embedderPromise) {
    console.log("开始加载本地模型 bge-small-zh-v1.5");
    embedderPromise = pipeline(
      "feature-extraction",
      "Xenova/bge-small-zh-v1.5",
      {
        quantized: true,
        progress_callback: (info: any) => {
          console.log("模型加载进度", info);
        },
      } as any,
    );
  }
  return embedderPromise;
}

/**
 * 批量文档chunk向量化（文档 passage: 前缀）
 * 分批并发，限制每批最多4条，避免浏览器卡死
 */
export async function embedPassageTexts(texts: string[], batchSize = 4) {
  const embedder = await getEmbedder();
  const embeddings: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batchTexts = texts.slice(i, i + batchSize);

    // 关键改动：回调显式标注 Promise<number[]>
    const batchPromises: Promise<number[]>[] = batchTexts.map(
      async (text): Promise<number[]> => {
        const output = await embedder(`passage: ${text}`, {
          pooling: "mean",
          normalize: true,
        });
        return Array.from(output.data);
      },
    );

    const batchResult = await Promise.all(batchPromises); // 现在类型是 number[][]
    embeddings.push(...batchResult);
  }
  return embeddings;
}

// 单个查询向量化（query: 前缀）
// 单个查询向量化（query: 前缀）
export async function embedQueryText(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(`query: ${text}`, {
    pooling: "mean",
    normalize: true,
  });
  return Array.from(output.data) as number[];
}
