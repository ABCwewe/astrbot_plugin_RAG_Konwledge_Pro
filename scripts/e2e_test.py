"""End-to-end RAG test against real services (Qdrant + NVIDIA embedding +
SiliconFlow reranker). Uses 剧情总结.md as the knowledge base.

Run:
    python scripts/e2e_test.py

Verifies: ingest → READY manifest → search+rerank → incremental update →
config-change rebuild → atomic version switch.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import RAGConfig  # noqa: E402
from core.engine import RAGEngine  # noqa: E402

# e2e 数据同样放在 AstrBot 的 plugin_data 专属目录，不落插件本体。
DATA_ROOT = (
    ROOT.parent.parent / "plugin_data" / "astrbot_plugin_RAG_Konwledge_Pro" / "e2e"
)
TEST_FILE = ROOT / "剧情总结.md"
CONFIG_FILE = ROOT / "e2e_config.json"

QUERIES = [
    "今州的岁主是谁？",
    "黑海岸的守岸人是谁？",
    "七丘篇的总督奥古斯塔有什么往事？",
    "弗洛洛和残星会在剧情里做了什么？",
]


def _load_config() -> RAGConfig:
    return RAGConfig.load(CONFIG_FILE)


async def _show_search(engine: RAGEngine, kb: str, query: str) -> None:
    results = await engine.search(kb, query)
    print(f"\n=== 查询: {query} ===")
    if not results:
        print("  (无结果)")
        return
    for r in results:
        score = (
            f"{r.vector_score:.4f}/rerank {r.rerank_score:.4f}"
            if r.rerank_score is not None
            else f"{r.vector_score:.4f}"
        )
        src = r.metadata.get("source", "")
        page = r.metadata.get("page")
        snippet = (r.content or r.image_path or "")[:70].replace("\n", " ")
        print(f"  [{score}] {src}{f' p{page}' if page else ''} :: {snippet}")


async def main() -> int:
    if not TEST_FILE.exists():
        print(f"缺少测试文件: {TEST_FILE}")
        return 1
    config = _load_config()
    config.validate()
    print("配置校验通过")

    # Fresh data root for a clean run.
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    engine = RAGEngine(config, DATA_ROOT)
    try:
        # 1. ingest → build v1
        print("\n[1] 导入 剧情总结.md 并构建索引…")
        result = await engine.ingest("default", [TEST_FILE])
        print("    ingest:", json.dumps(result, ensure_ascii=False))

        st = await engine.status("default")
        v = st["versions"].get(st["active_version"], {})
        print(
            f"    活动版本 v{st['active_version']} "
            f"{v.get('status')} 文档 {v.get('document_count')} 块 {v.get('chunk_count')}"
        )
        assert st["active_version"] == 1
        assert v.get("status") == "READY"
        assert v.get("document_count", 0) >= 1
        assert v.get("chunk_count", 0) >= 1
        print("    ✓ 构建成功")

        # 2. searches
        print("\n[2] 检索测试")
        for q in QUERIES:
            await _show_search(engine, "default", q)

        # 3. incremental update: modify the file inside the KB docs dir
        print("\n[3] 增量更新测试（修改文件后同步）")
        docs_file = engine.docs_dir("default") / "剧情总结.md"
        original = docs_file.read_text(encoding="utf-8")
        try:
            extra = "\n---\n## 99 测试补充章节\n- 剧情线：测试篇\n- 主角：测试角色\n\n剧情摘要：这是用于验证增量更新的补充章节内容。\n"
            docs_file.write_text(original + extra, encoding="utf-8")
            sync_result = await engine.sync("default")
            print("    sync:", json.dumps(sync_result, ensure_ascii=False))
            assert sync_result["updated"] == 1, "期望文件变化被识别为 updated"
            st = await engine.status("default")
            v = st["versions"][st["active_version"]]
            print(f"    更新后: 文档 {v['document_count']} 块 {v['chunk_count']}")
            print("    ✓ 增量更新生效")
        finally:
            docs_file.write_text(original, encoding="utf-8")

        # 4. config change → automatic rebuild with atomic switch.
        # Simulated as the engine being restarted with a changed config.
        print("\n[4] 配置变更自动重建测试")
        new_config = RAGConfig.from_dict(
            {
                **json.loads(CONFIG_FILE.read_text(encoding="utf-8")),
                "chunking": {"separator": "\n\n", "chunk_size": 400, "chunk_overlap": 50},
            }
        )
        engine2 = RAGEngine(new_config, DATA_ROOT)
        try:
            rebuild_result = await engine2.ingest("default", [TEST_FILE])
        finally:
            await engine2.close()
        print("    rebuild:", json.dumps(rebuild_result, ensure_ascii=False))
        assert rebuild_result["action"] == "rebuilt"
        assert rebuild_result["version"] == 2
        st = await engine.status("default")
        assert st["active_version"] == 2
        assert "astrbot_rag_default_v1" not in {
            m["collection"] for m in st["versions"].values()
        }
        print("    ✓ 自动重建并切换到 v2，旧 collection 已清理")

        # 5. search on the new version still works
        print("\n[5] 重建后检索验证")
        await _show_search(engine, "default", QUERIES[0])

        cache_stats = await engine._cache.stats()
        print(f"\n缓存统计: {json.dumps(cache_stats)}")
        print("\n全部通过 ✅")
        return 0
    finally:
        await engine.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
