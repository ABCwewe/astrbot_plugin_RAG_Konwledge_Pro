"""AstrBot integration adapter (AGENTS.md §31-§32).

Bridges the AstrBot plugin layer (config, events, messages, WebUI) to the
self-contained :class:`~core.engine.RAGEngine`. All AstrBot-specific API
usage is confined here; the engine never sees AstrBot objects.

The plugin's persistent data lives under ``data/plugin_data/astrbot_rag``
(AGENTS.md §35 persistence rule), not inside the plugin directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from astrbot.api import logger as astrbot_logger
from astrbot.core.star import StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:  # loaded as part of the plugin package inside AstrBot
    from ..core import RAGConfig, RAGEngine
    from ..core.exceptions import RAGError
except ImportError:  # standalone (tests / scripts)
    from core import RAGConfig, RAGEngine
    from core.exceptions import RAGError

logger = logging.getLogger("rag.adapter")

#: 插件在 data/plugin_data/ 下的独立目录名（与 metadata.yaml name 一致）。
PLUGIN_NAME = "astrbot_plugin_RAG_Konwledge_Pro"


def _bridge_rag_logs() -> None:
    """Forward ``rag.*`` std-logging (engine modules) into AstrBot's plugin
    logger so all logs surface in one place (AGENTS.md §35)."""
    rag_logger = logging.getLogger("rag")
    astrbot_log = astrbot_logger

    class _ForwardHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = (record.levelname or "info").lower()
                getattr(astrbot_log, level, astrbot_log.info)(self.format(record))
            except Exception:  # pragma: no cover - logging must never break flow
                pass

    if not any(isinstance(h, _ForwardHandler) for h in rag_logger.handlers):
        handler = _ForwardHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        rag_logger.addHandler(handler)


def _fmt_result(result: dict) -> str:
    return (
        f"新增 {result.get('added', 0)}，更新 {result.get('updated', 0)}，"
        f"未变化 {result.get('unchanged', 0)}，删除 {result.get('deleted', 0)}，"
        f"块总数 {result.get('chunks', 0)}"
    )


class AstrBotRAGAdapter:
    def __init__(self, context, config: dict | None = None) -> None:
        self.context = context
        self._cfg = config or {}
        self._engine: RAGEngine | None = None
        self.default_kb = str(self._cfg.get("default_kb_id", "default"))

    # -- lifecycle --------------------------------------------------------

    async def initialize(self) -> None:
        _bridge_rag_logs()
        try:
            rag_config = self._build_rag_config()
            rag_config.validate()
        except RAGError as exc:
            logger.warning("[RAG] 配置未完成，引擎暂不启动: %s", exc)
            astrbot_logger.warning("[RAG] 配置未完成，请在插件设置中填写 Embedding/Rerank 配置后重启: %s", exc)
            return
        # 所有运行时文件（索引、缓存、临时文件）统一放在
        # data/plugin_data/astrbot_plugin_RAG_Konwledge_Pro/ 下，
        # 与插件本体目录完全隔离（AGENTS.md §35）。
        try:
            data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("[RAG] StarTools.get_data_dir 失败，回退到 plugin_data 根: %s", exc)
            data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
            data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._tmp_dir = data_dir / "tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._engine = RAGEngine(rag_config, data_dir)
        astrbot_logger.info(
            "[RAG] 引擎已初始化 (kb=%s, qdrant=%s, data=%s)",
            self.default_kb,
            rag_config.qdrant.url,
            data_dir,
        )

    async def terminate(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None

    def _require_engine(self) -> RAGEngine:
        if self._engine is None:
            raise RuntimeError("RAG 引擎未初始化：请在插件设置中填写 Embedding/Rerank/Qdrant 配置后重启插件")
        return self._engine

    @property
    def tmp_dir(self) -> Path:
        """插件专属临时目录（data/plugin_data/<plugin>/tmp/）。"""
        return self._tmp_dir

    # -- config -----------------------------------------------------------

    def _build_rag_config(self) -> RAGConfig:
        cfg = self._cfg
        return RAGConfig.from_dict(
            {
                "kb_id": self.default_kb,
                "qdrant": {
                    "url": cfg.get("qdrant_url", "http://127.0.0.1:6333"),
                    "api_key": cfg.get("qdrant_api_key") or None,
                    "timeout": float(cfg.get("qdrant_timeout", 60)),
                },
                "embedding": {
                    "api_base": cfg.get("embedding_api_base", ""),
                    "api_key": cfg.get("embedding_api_key", ""),
                    "model": cfg.get("embedding_model", ""),
                    "dimension": int(cfg.get("embedding_dimension", 0) or 0),
                    "batch_size": int(cfg.get("embedding_batch_size", 32)),
                    "concurrency": int(cfg.get("embedding_concurrency", 4)),
                    "extra_params": {
                        "input_type": cfg.get("embedding_input_type", "passage"),
                        "truncate": cfg.get("embedding_truncate", "END"),
                    },
                },
                "image": {
                    "enabled": bool(cfg.get("image_enabled", False)),
                    "api_base": cfg.get("image_api_base"),
                    "api_key": cfg.get("image_api_key"),
                    "model": cfg.get("image_model"),
                    "dimension": int(cfg["image_dimension"]) if cfg.get("image_dimension") else None,
                    "search_always": bool(cfg.get("image_search_always", False)),
                },
                "rerank": {
                    "enabled": bool(cfg.get("rerank_enabled", True)),
                    "api_base": cfg.get("rerank_api_base", ""),
                    "api_key": cfg.get("rerank_api_key", ""),
                    "model": cfg.get("rerank_model", ""),
                },
                "chunking": {
                    "separator": cfg.get("chunk_separator", "\n\n"),
                    "chunk_size": int(cfg.get("chunk_size", 800)),
                    "chunk_overlap": int(cfg.get("chunk_overlap", 100)),
                },
                "top_k": int(cfg.get("top_k", 30)),
                "top_n": int(cfg.get("top_n", 6)),
            }
        )

    # -- operations (dict for WebUI, text for chat commands) --------------

    async def search(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        include_images: bool | None = None,
    ):
        return await self._require_engine().search(
            kb_id,
            query,
            top_k=top_k,
            top_n=top_n,
            include_images=include_images,
        )

    async def search_text(self, kb_id: str, query: str) -> str:
        results = await self.search(kb_id, query)
        if not results:
            return "未检索到相关内容。"
        context = RAGEngine.format_context(results)
        scores = ", ".join(
            f"{r.vector_score:.4f}" + (f"/{r.rerank_score:.4f}" if r.rerank_score is not None else "")
            for r in results
        )
        return f"检索到 {len(results)} 条相关内容：\n\n{context}\n\n（分数: {scores}）"

    async def ingest(self, kb_id: str, paths: list[str]) -> dict:
        result = await self._require_engine().ingest(kb_id, paths)
        return {**result, "kb_id": kb_id}

    async def ingest_text(self, kb_id: str, paths: list[str]) -> str:
        result = await self.ingest(kb_id, paths)
        return f"导入完成。{_fmt_result(result)}"

    async def rebuild(self, kb_id: str) -> dict:
        return await self._require_engine().rebuild(kb_id)

    async def rebuild_text(self, kb_id: str) -> str:
        result = await self.rebuild(kb_id)
        return (
            f"重建完成：新版本 v{result['version']}，"
            f"文档 {result['documents']} 个，块 {result['chunks']} 个。"
        )

    async def status_dict(self, kb_id: str) -> dict:
        status = await self._require_engine().status(kb_id)
        status["kb_id"] = kb_id
        return status

    async def status_text(self, kb_id: str) -> str:
        st = await self.status_dict(kb_id)
        lines = [f"知识库: {kb_id}"]
        lines.append(f"活动版本: v{st['active_version']}" if st["active_version"] else "活动版本: 无")
        if st["config_changed"]:
            lines.append("⚠ 配置已变更，需要重建索引")
        for version, m in st.get("versions", {}).items():
            lines.append(
                f"  v{version}: {m['status']} (文档 {m['document_count']}, 块 {m['chunk_count']})"
                + (f" 错误: {m['error']}" if m.get("error") else "")
            )
        progress = st.get("progress")
        if progress and progress.get("status") not in ("idle", "READY"):
            lines.append(
                f"  构建中: {progress['status']} "
                f"{progress['processed_documents']}/{progress['total_documents']} 文档, "
                f"{progress['processed_chunks']} 块"
                + (f" 错误: {progress['error']}" if progress.get("error") else "")
            )
        return "\n".join(lines)

    async def list_kbs(self) -> list[dict]:
        engine = self._require_engine()
        out = []
        for kb_id in await engine.list_kbs():
            st = await engine.status(kb_id)
            active = st.get("active_version")
            version = st.get("versions", {}).get(active, {}) if active else {}
            out.append(
                {
                    "kb_id": kb_id,
                    "active_version": active,
                    "status": version.get("status", "none") if active else "none",
                    "document_count": version.get("document_count", 0),
                    "chunk_count": version.get("chunk_count", 0),
                }
            )
        return out

    async def list_text(self) -> str:
        kbs = await self.list_kbs()
        if not kbs:
            return "暂无知识库。使用 WebUI 或 /rag ingest 上传文档。"
        lines = ["已有知识库:"]
        for kb in kbs:
            lines.append(
                f"  {kb['kb_id']}: v{kb['active_version']} "
                f"{kb['status']} (文档 {kb['document_count']}, 块 {kb['chunk_count']})"
            )
        return "\n".join(lines)

    async def llm_search(self, query: str) -> str | None:
        """LLM tool entry point: returns RAG context or None (no noise)."""
        try:
            results = await self.search(self.default_kb, query)
        except RAGError as exc:
            logger.warning("[RAG][LLM] 检索失败: %s", exc)
            return None
        if not results:
            return None
        return RAGEngine.format_context(results)

    async def get_build_progress_dict(self, kb_id: str) -> dict | None:
        progress = self._require_engine().get_build_progress(kb_id)
        return progress.to_dict() if progress else None
