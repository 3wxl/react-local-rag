import { embedQueryText } from "./embedding";
import { cosineSimilarity } from "./similarity";
import type { TextChunk } from "../types/doc";

export interface VectorChunk extends TextChunk {
  vector: number[];
}

export async function searchRelevant(
  query: string,
  chunks: VectorChunk[],
  topK = 3,
): Promise<{ content: string; score: number }[]> {
  const queryVec = await embedQueryText(query);
  const scored = chunks.map((item) => ({
    content: item.content,
    score: cosineSimilarity(queryVec, item.vector),
  }));
  scored.sort((x, y) => y.score - x.score);
  return scored.slice(0, topK);
}
