"""AstrBot integration adapter (AGENTS.md §31-§32).

Bridges the AstrBot plugin layer (config, events, messages, WebUI) to the
self-contained :class:`~core.engine.RAGEngine`. All AstrBot-specific API
usage is confined here; the engine never sees AstrBot objects.

The plugin's persistent data lives under
``data/plugin_data/astrbot_plugin_RAG_Konwledge_Pro`` (AGENTS.md §35
persistence rule), never inside the plugin directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from astrbot.api import logger as astrbot_logger
from astrbot.core.star import StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:  # loaded as part of the plugin package inside AstrBot
    from ..core import RAGConfig, RAGEngine
    from ..core.exceptions import ConfigurationError, RAGError
except ImportError:  # standalone (tests / scripts)
    from core import RAGConfig, RAGEngine
    from core.exceptions import ConfigurationError, RAGError

from .config_utils import (
    apply_patch,
    group_members,
    group_order,
    load_search_defaults,
    mask_config,
    save_search_defaults,
)
from .image_utils import find_message_image

logger = logging.getLogger("rag.adapter")

#: 插件在 data/plugin_data/ 下的独立目录名（与 metadata.yaml name 一致）。
PLUGIN_NAME = "astrbot_plugin_RAG_Konwledge_Pro"

_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "_conf_schema.json"


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
        self._data_dir: Path | None = None
        self._tmp_dir: Path | None = None
        self.default_kb = str(self._cfg.get("default_kb_id", "default"))
        self._selected_kb: str | None = None

    @property
    def current_kb(self) -> str:
        """知识库上传目标：WebUI 显式选择优先，否则默认知识库。"""
        return self._selected_kb or self.default_kb

    def select_kb(self, kb_id: str) -> str:
        """Set the WebUI's current knowledge base (upload target)."""
        kb_id = (kb_id or "").strip()
        self._selected_kb = kb_id or None
        return self.current_kb

    # -- lifecycle --------------------------------------------------------

    async def initialize(self) -> None:
        _bridge_rag_logs()
        try:
            await self._restart_engine()
        except RAGError as exc:
            logger.warning("[RAG] 配置未完成，引擎暂不启动: %s", exc)
            astrbot_logger.warning(
                "[RAG] 配置未完成，请在插件设置或 WebUI 中填写 Embedding/Rerank 配置后重试: %s",
                exc,
            )

    async def terminate(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None

    async def _restart_engine(self) -> None:
        """(Re)build the engine from the current config. Safe to call when
        config changes at runtime (WebUI) — closes the old engine first."""
        rag_config = self._build_rag_config()
        rag_config.validate()
        if self._data_dir is None:
            # 所有运行时文件（索引、缓存、临时文件）统一放在
            # data/plugin_data/astrbot_plugin_RAG_Konwledge_Pro/ 下，
            # 与插件本体目录完全隔离（AGENTS.md §35）。
            try:
                data_dir = StarTools.get_data_dir(PLUGIN_NAME)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "[RAG] StarTools.get_data_dir 失败，回退到 plugin_data 根: %s", exc
                )
                data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
                data_dir.mkdir(parents=True, exist_ok=True)
            self._data_dir = data_dir
            self._tmp_dir = data_dir / "tmp"
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
        old_engine = self._engine
        new_engine = RAGEngine(rag_config, self._data_dir)
        if old_engine is not None:
            await old_engine.close()
        self._engine = new_engine
        self.default_kb = str(self._cfg.get("default_kb_id", "default"))
        astrbot_logger.info(
            "[RAG] 引擎已就绪 (kb=%s, qdrant=%s, data=%s)",
            self.default_kb,
            rag_config.qdrant.url,
            self._data_dir,
        )

    def _require_engine(self) -> RAGEngine:
        if self._engine is None:
            raise RuntimeError("RAG 引擎未初始化：请在插件设置中填写 Embedding/Rerank/Qdrant 配置后重启插件")
        return self._engine

    @property
    def tmp_dir(self) -> Path:
        """插件专属临时目录（data/plugin_data/<plugin>/tmp/）。

        访问时确保目录存在——目录可能被外部清理，缺失时重建，避免
        mkdtemp 抛 FileNotFoundError。
        """
        if self._tmp_dir is None:
            raise RuntimeError("RAG 引擎未初始化：请在插件设置或 WebUI 配置页完成配置")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
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
                    "auto_search": bool(cfg.get("image_auto_search", True)),
                    "min_score": float(cfg.get("image_min_score", 0.0) or 0.0),
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

    async def search_multi(
        self,
        kb_ids: list[str],
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        include_images: bool | None = None,
    ):
        return await self._require_engine().search_multi(
            kb_ids,
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
        if not self.default_kb:
            # default_kb_id 留空 = 禁用默认知识库：工具静默跳过
            return None
        try:
            results = await self.search(self.default_kb, query)
        except RAGError as exc:
            logger.warning("[RAG][LLM] 检索失败: %s", exc)
            return None
        if not results:
            return None
        return RAGEngine.format_context(results)

    async def auto_image_search_context(self, event) -> str | None:
        """消息带图时自动图片向量检索，返回注入用上下文。

        - 仅当 image.enabled 且 image.auto_search 时生效
        - 只读取事件中的图片（正文优先，其次引用消息），不修改事件
        - 无图 / 检索无结果 / 任何失败都返回 None（静默跳过，绝不干扰
          AstrBot 原生的图像转述与直通流程）
        """
        try:
            engine = self._require_engine()
            if not engine.config.image.enabled or not engine.config.image.auto_search:
                return None
            if not self.default_kb:
                # default_kb_id 留空 = 禁用默认知识库：自动图片检索静默跳过
                return None
        except RuntimeError:
            return None
        get_messages = getattr(event, "get_messages", None)
        messages = get_messages() if get_messages else []
        comp = find_message_image(messages)
        if comp is None:
            return None
        try:
            converter = getattr(comp, "convert_to_file_path", None)
            image_path = (
                await converter() if converter else (comp.url or comp.file or "")
            )
            if not image_path:
                return None
            results = await engine.search_image_by_path(
                self.default_kb, image_path, top_n=3
            )
        except Exception as exc:
            logger.debug("[RAG] 自动图片检索跳过: %s", exc)
            return None
        if not results:
            return None
        return RAGEngine.format_context(results)

    async def get_build_progress_dict(self, kb_id: str) -> dict | None:
        progress = self._require_engine().get_build_progress(kb_id)
        return progress.to_dict() if progress else None

    # -- config management (WebUI) ----------------------------------------

    def _load_schema(self) -> dict:
        try:
            schema = json.loads(_SCHEMA_FILE.read_text(encoding="utf-8-sig"))
            return schema if isinstance(schema, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.exception("[RAG] 读取 _conf_schema.json 失败")
            return {}

    async def get_config_payload(self) -> dict:
        """Schema + current values (secrets masked) for the WebUI form."""
        schema = self._load_schema()
        return {
            "schema": schema,
            "config": mask_config(dict(self._cfg)),
            "groups": group_order(schema),
            "group_members": group_members(schema),
            "default_kb": self.default_kb,
        }

    async def update_config(self, patch: dict) -> dict:
        """Apply a validated patch from the WebUI, persist it, restart engine.

        Raises:
            ConfigurationError: when the merged config fails validation
                (nothing is persisted in that case).
        """
        schema = self._load_schema()
        if not isinstance(patch, dict):
            raise ConfigurationError("配置数据格式错误")
        merged = apply_patch(self._cfg, patch, schema)

        # Validate BEFORE persisting: build a config from the merged values.
        old_cfg = self._cfg
        self._cfg = merged
        try:
            rag_config = self._build_rag_config()
            rag_config.validate()
        except Exception:
            self._cfg = old_cfg  # roll back in-memory state
            raise
        await self._restart_engine()
        # Persist last: if persistence fails the engine still runs on the new
        # in-memory config; log and surface the error to the caller.
        saver = getattr(old_cfg, "save_config", None) if isinstance(old_cfg, dict) else None
        if saver is not None:
            saver(merged)
        return await self.get_config_payload()

    # -- knowledge base management (WebUI) --------------------------------

    async def delete_kb(self, kb_id: str) -> None:
        await self._require_engine().delete_kb(kb_id)

    async def drop_index(self, kb_id: str) -> None:
        """Delete only the KB index; documents and the KB itself are kept."""
        await self._require_engine().drop_index(kb_id)

    async def create_kb(self, kb_id: str) -> dict:
        if not kb_id or not kb_id.strip():
            from ..core.exceptions import ConfigurationError

            raise ConfigurationError("知识库 ID 不能为空")
        return await self._require_engine().create_kb(kb_id.strip())

    # -- 默认聚合检索知识库 ------------------------------------------------

    def _require_data_dir(self) -> Path:
        if self._data_dir is None:
            raise RuntimeError("RAG 引擎未初始化：请先在配置页完成 Embedding/Rerank 配置")
        return self._data_dir

    def _defaults_path(self) -> Path:
        return self._require_data_dir() / "search_defaults.json"

    def get_default_search_kbs(self) -> list[str]:
        """Persisted KB ids enabled by default for aggregated search."""
        return load_search_defaults(self._defaults_path())

    def set_default_search_kbs(self, kb_ids: list[str]) -> list[str]:
        return save_search_defaults(self._defaults_path(), kb_ids)

    async def list_documents(self, kb_id: str) -> list[dict]:
        return await self._require_engine().list_documents(kb_id)

    async def delete_document(self, kb_id: str, filename: str) -> dict:
        return await self._require_engine().remove_document(kb_id, filename)
