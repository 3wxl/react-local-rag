/**
 * 全局推理锁：带优先级的互斥队列。
 *
 * 浏览器 ONNX Runtime (WASM) 在共享堆上分配张量并执行推理，
 * BGE 向量化/检索/校验 与本地 LLM (split/evaluate/summarize/generate)
 * 两路 session 同时运行会抢占 WASM 堆，出现 std::bad_alloc / ERROR_CODE: 6。
 *
 * 业务优先级：
 * - "high"（默认）：用户实时问答链路上的推理（检索 / 生成 / 幻觉校验 / 子问题拆分等）。
 *   用户正在等待回复，必须最先拿到锁。
 * - "low"：后台任务（空闲时的历史摘要压缩、文档批量向量化）。
 *   可以延后，队列中只要还有 high 等待者就持续让路。
 *
 * 调度规则：每次锁释放时，先放行队列中的 high 等待者；没有 high 才放行 low。
 * 注意：已经持有的推理不可抢占（ORT 调用无法中断），优先级只影响「排队顺序」；
 * 因此批量向量化在调用方按小批（8 条）分次持锁，使问答推理能在批次间插入。
 *
 * 用法：
 *   const release = await acquireInferenceLock('llm:summarize', 'low');
 *   try {
 *     await runInference();
 *   } finally {
 *     release();
 *   }
 */

export type InferencePriority = "high" | "low";//`InferencePriority`：优先级枚举，只有`high`/`low`

interface Waiter {//等待锁的任务对象
  priority: InferencePriority;//`priority`：任务优先级
  grant: () => void;//`grant`：放行函数，用于释放锁
}

let held = false; // 布尔标记：true=锁当前被占用；false=空闲
const waiters: Waiter[] = []; // 等待队列，存放所有排队等待拿锁的任务


/** 释放后尝试放行下一个等待者：high 优先，无 high 时才取队首 low */
function pump(): void {
  if (held || waiters.length === 0) return;
  const highIdx = waiters.findIndex((w) => w.priority === "high");
  const picked =
    highIdx >= 0
      ? waiters.splice(highIdx, 1)[0]
      : waiters.shift()!;
  held = true;
  picked.grant();
}

/**
 * 请求推理锁。返回 release 函数；调用方在推理结束（成功或失败）后必须调用 release()，
 * 否则后续所有推理任务会永久阻塞。
 *
 * @param taskName 仅用于调试日志，标识是哪一类推理在持有锁
 * @param priority "high"=实时问答（默认）；"low"=后台任务，给 high 让路
 */
export function acquireInferenceLock(
  taskName: string,
  priority: InferencePriority = "high",
): Promise<() => void> {
  return new Promise<() => void>((resolve) => {
    let released = false;
    const waiter: Waiter = {
      priority,
      grant: () => {
        console.debug(
          `[inference-lock] acquired: ${taskName} (${priority}, waiting=${waiters.length})`,
        );
        resolve(() => {
          // 幂等：重复 release 不影响后续队列
          if (released) return;
          released = true;
          held = false;
          pump();
        });
      },
    };
    waiters.push(waiter);
    pump();
  });
}
