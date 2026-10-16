#!/usr/bin/env python
"""把历史 doc_id（`uuid5(name:content_hash)`）迁移为稳定口径（`uuid5(name)`）。

用法（在 apps/api 目录下）：
    uv run python scripts/migrate_doc_ids.py --dry-run   # 先看会做什么
    uv run python scripts/migrate_doc_ids.py             # 执行

背景（为什么必须迁）：
旧实现的 doc_id 把**内容哈希**并进了身份，于是"改一次文档内容"会派生出**新的 doc_id**，
"同名替换"分支永远走不到 —— 知识库里会同时留着同一份文档的多个历史版本，
旧版本继续参与检索（用户看到的是过期内容）。内容哈希的正确位置是版本属性，
不是主键。

迁移规则（**幂等**，可重复执行）：
1. 按 doc_name 分组，目标 id = uuid5("rag-doc:" + name)
2. 每组保留"最新的一份"（未删除优先，其次 created_at 最新），把它的文档行与块迁到目标 id
3. 同组的其它历史版本：删块与向量 + 软删除文档行（保留审计痕迹，不物理删除）
4. 已经符合新口径的文档原样跳过
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.services.ingest_service import _doc_id_for  # noqa: E402
from app.store import make_store  # noqa: E402


async def migrate(dry_run: bool) -> None:
    store = make_store(settings.database_url, settings.data_dir)
    await store.init()
    try:
        documents = await store.list_documents(include_deleted=True)
        groups: dict[str, list] = {}
        for doc in documents:
            groups.setdefault(doc.doc_name, []).append(doc)

        moved = merged = skipped = 0
        for name, docs in groups.items():
            target = _doc_id_for(name)
            alive = [d for d in docs if d.deleted_at is None]
            pool = alive or docs
            # 未删除优先，其次 created_at 最新 —— 保留用户当前看到的那一份
            survivor = sorted(pool, key=lambda d: d.created_at or "", reverse=True)[0]
            losers = [d for d in docs if d.doc_id != survivor.doc_id]

            if survivor.doc_id != target:
                print(f"[迁移] {name}: {survivor.doc_id} → {target}")
                if not dry_run:
                    await store.rename_document(survivor.doc_id, target)
                moved += 1
            else:
                skipped += 1

            for loser in losers:
                print(f"  [归并] 历史版本 {loser.doc_id}（created_at={loser.created_at}）→ 软删除")
                if not dry_run:
                    # 先删块与向量：历史版本的内容不该再参与检索
                    await store.soft_delete_doc(loser.doc_id)
                    remaining = [
                        c
                        for c in await store.load_chunks(loser.index_version or "", filters=None)
                        if c.doc_id == loser.doc_id
                    ]
                    if remaining:
                        await store.replace_doc_chunks(loser.doc_id, loser.index_version, [])
                merged += 1

        print(
            f"\n完成：迁移 {moved} 篇 / 归并历史版本 {merged} 个 / 已符合新口径 {skipped} 篇"
            + ("（dry-run，未写入）" if dry_run else "")
        )
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="doc_id 口径迁移（幂等）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写库")
    args = parser.parse_args()
    asyncio.run(migrate(args.dry_run))


if __name__ == "__main__":
    main()
