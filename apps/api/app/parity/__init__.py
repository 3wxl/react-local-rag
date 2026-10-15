"""跨语言同口径层（parity）。

本包是浏览器端 Lite 模式（apps/web/src/utils）在服务端的**逐条移植**，
目标是「同一份输入产出同一份输出」，而不是「重新实现一个更好的版本」。

口径由 packages/chunker-spec/SPEC.md 定义，由 apps/api/tests/test_*parity*.py 强制：
任何一次改动只要让两端的 contentHash / 排名 / 分数不再一致，CI 就红。

约定：
- 纯函数、无 I/O、不 import FastAPI（便于单测与命令行脚本复用）
- 默认参数一律照搬 TS 常量，不"顺手优化"（改一个常量就是改索引口径）
- JS 语义要显式还原：UTF-16 码元哈希、`Math.imul` 的 32 位环绕、`undefined` → `null`、
  排序稳定性、`Map` 的插入序等（注释里逐条标明出处）
"""
