"""FNV-1a 32 位哈希（与 apps/web/src/utils/chunk.ts 的 fnv1a32 逐位对齐）。

JS 原实现按 **UTF-16 码元** 迭代（`charCodeAt`），不是按 Unicode 码点：

    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, "0");

因此 Python 侧必须先 `encode("utf-16-le")` 再按 2 字节一组读无符号短整型。
若按码点迭代，非 BMP 字符（emoji、𝄞 等，UTF-16 里是代理对）会算出**不同的哈希** ——
contentHash 不等 → 跨端索引对不上。fixtures/nonbmp.md 专门覆盖这种情况。
"""

from __future__ import annotations

FNV_OFFSET_BASIS = 0x811C9DC5
FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def fnv1a32(text: str) -> str:
    """32 位 FNV-1a，返回 8 位十六进制字符串（与 TS 完全一致）。"""
    h = FNV_OFFSET_BASIS
    units = text.encode("utf-16-le")
    for i in range(0, len(units), 2):
        unit = units[i] | (units[i + 1] << 8)
        h ^= unit
        # 对应 Math.imul：32 位环绕乘法（取模 2**32 与 JS 的有符号实现等价）
        h = (h * FNV_PRIME) & _MASK32
    return format(h, "08x")
