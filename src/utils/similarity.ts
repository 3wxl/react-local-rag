/**
 * 两个向量的点积（内积）
 * 例: [1,2,3]·[4,5,6] = 1*4+2*5+3*6 = 32
 */
function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * 余弦相似度（优化版）
 * 当前embedding全部开启 normalize:true，向量模长恒=1，余弦相似度直接等于点积
 * 结果范围 [-1, 1]，越接近1越相似
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  // 增加安全判断，防止空向量
  if (a.length === 0 || b.length === 0) return 0;
  return dotProduct(a, b);
}
