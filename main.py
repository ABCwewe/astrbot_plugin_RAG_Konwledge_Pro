"""AstrBot RAG 插件入口 (AGENTS.md §31).

Only AstrBot lifecycle wiring lives here — chunking/embedding/qdrant/indexing
stay in ``core/`` and are reached through the adapter.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request
from astrbot.core.star.filter.command import GreedyStr

from .adapter import AstrBotRAGAdapter

ROUTE_PREFIX = "astrbot_plugin_RAG_Konwledge_Pro"
"""Web API 路由前缀。前端 bridge 会把插件的 yaml name 前置到请求路径上，
因此路由必须注册为 ``/<插件yaml名>/<suffix>``（与其他插件一致）。"""

_SUPPORTED_EXTS = {
    ".txt", ".md", ".markdown", ".text", ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
}


class RAGPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self.adapter = AstrBotRAGAdapter(context, self.config)

    async def initialize(self) -> None:
        await self.adapter.initialize()
        ctx = self.context
        ctx.register_web_api(f"/{ROUTE_PREFIX}/status", self.web_status, ["GET"], "RAG 索引状态")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/list", self.web_list, ["GET"], "RAG 知识库列表")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/search", self.web_search, ["GET"], "RAG 检索")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/ingest", self.web_ingest, ["POST"], "RAG 导入文档")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/rebuild", self.web_rebuild, ["POST"], "RAG 重建索引")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/progress", self.web_progress, ["GET"], "RAG 构建进度")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/config", self.web_config_get, ["GET"], "RAG 配置读取")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/config", self.web_config_update, ["POST"], "RAG 配置保存")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/delete", self.web_kb_delete, ["POST"], "RAG 删除知识库")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/create", self.web_kb_create, ["POST"], "RAG 新建知识库")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/select", self.web_kb_select, ["POST"], "RAG 选择当前知识库")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/index/delete", self.web_kb_index_delete, ["POST"], "RAG 删除知识库索引")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/defaults", self.web_kb_defaults_get, ["GET"], "RAG 默认聚合检索知识库")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/defaults", self.web_kb_defaults_set, ["POST"], "RAG 设置默认聚合检索知识库")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/documents", self.web_kb_documents, ["GET"], "RAG 知识库文档列表")
        ctx.register_web_api(f"/{ROUTE_PREFIX}/kb/document/delete", self.web_kb_document_delete, ["POST"], "RAG 删除知识库文档")
        logger.info("[RAG] 插件已加载，路由前缀 /%s", ROUTE_PREFIX)

    # ------------------------------------------------------------------
    # 管理命令
    # ------------------------------------------------------------------

    @filter.command("rag")
    async def rag_command(self, event: AstrMessageEvent, action: str = "status", query: GreedyStr = None):
        """RAG 知识库管理。用法: /rag list|status|rebuild|search <问题>"""
        try:
            kb = self.adapter.default_kb
            if action == "search":
                if not query:
                    yield event.plain_result("用法: /rag search <问题>")
                    return
                yield event.plain_result(await self.adapter.search_text(kb, query))
            elif action == "list":
                yield event.plain_result(await self.adapter.list_text())
            elif action == "rebuild":
                yield event.plain_result(await self.adapter.rebuild_text(kb))
            elif action == "ingest":
                yield event.plain_result(await self._ingest_from_message(event, kb))
            elif action == "status":
                yield event.plain_result(await self.adapter.status_text(kb))
            else:
                yield event.plain_result(
                    f"未知操作: {action}\n可用操作: /rag list /rag status /rag rebuild /rag ingest /rag search <问题>"
                )
        except Exception as exc:
            logger.exception("[RAG] 命令执行失败")
            yield event.plain_result(f"RAG 操作失败: {exc}")

    async def _ingest_from_message(self, event: AstrMessageEvent, kb: str) -> str:
        paths = []
        for component in event.get_messages():
            file_path = getattr(component, "file", None) or getattr(component, "path", None)
            if isinstance(file_path, str) and file_path:
                paths.append(file_path)
        paths = [p for p in paths if Path(p).suffix.lower() in _SUPPORTED_EXTS]
        if not paths:
            return "未在消息中找到可导入的文件（支持 txt/md/pdf/图片）。"
        return await self.adapter.ingest_text(kb, paths)

    # ------------------------------------------------------------------
    # LLM 工具：让模型在回答时按需检索知识库
    # ------------------------------------------------------------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """消息带图时自动注入图片向量检索结果。

        只追加 extra_user_content_parts（单轮生效），完全不触碰
        req.image_urls / system_prompt —— AstrBot 原生的图像直通
        （视觉模型）与图像转述（非视觉模型）流程不受影响。
        """
        try:
            context = await self.adapter.auto_image_search_context(event)
        except Exception as exc:
            logger.debug("[RAG] 自动图片检索跳过: %s", exc)
            return
        if not context:
            return
        from astrbot.core.agent.message import TextPart

        req.extra_user_content_parts.append(TextPart(text=context).mark_as_temp())
        logger.debug("[RAG] 已注入自动图片检索上下文 (%d 字符)", len(context))

    @filter.llm_tool(name="rag_search")
    async def rag_search_tool(self, event: AstrMessageEvent, query: str):
        """在增强 RAG 知识库中检索与问题最相关的内容片段，返回带来源标注的上下文。

        Args:
            query(string): 需要检索的问题或关键词。
        """
        try:
            return await self.adapter.llm_search(query)
        except Exception as exc:
            logger.exception("[RAG][LLM] 检索失败")
            return None

    # ------------------------------------------------------------------
    # Web API（供 WebUI 页面调用）
    # ------------------------------------------------------------------

    async def web_status(self):
        try:
            return json_response(await self.adapter.status_dict(self.adapter.default_kb))
        except Exception as exc:
            logger.exception("[RAG] web/status 失败")
            return error_response(str(exc))

    async def web_list(self):
        try:
            return json_response(await self.adapter.list_kbs())
        except Exception as exc:
            logger.exception("[RAG] web/list 失败")
            return error_response(str(exc))

    async def web_search(self):
        try:
            query = request.query.get("q", "")
            top_n = request.query.get("top_n", None, type=int)
            kb_ids_raw = request.query.get("kb_ids", "")
            kb_ids = [k.strip() for k in kb_ids_raw.split(",") if k.strip()]
            if kb_ids:
                results = await self.adapter.search_multi(kb_ids, query, top_n=top_n)
            else:
                defaults = self.adapter.get_default_search_kbs()
                if defaults:
                    results = await self.adapter.search_multi(defaults, query, top_n=top_n)
                else:
                    kb = request.query.get("kb_id", self.adapter.default_kb)
                    results = await self.adapter.search(kb, query, top_n=top_n)
            return json_response(
                {
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "document_id": r.document_id,
                            "content": r.content,
                            "image_path": r.image_path,
                            "vector_score": r.vector_score,
                            "rerank_score": r.rerank_score,
                            "metadata": r.metadata,
                        }
                        for r in results
                    ]
                }
            )
        except Exception as exc:
            logger.exception("[RAG] web/search 失败")
            return error_response(str(exc))

    async def web_ingest(self):
        try:
            # 上传接口无法携带查询参数（bridge 限制），目标知识库由
            # kb/select 设置（request.query 仍兼容旧调用方）。
            kb = request.query.get("kb_id") or self.adapter.current_kb
            files = await request.files()
            if not files:
                return error_response("未收到文件", status_code=400)
            # 先落到插件专属临时目录（data/plugin_data/<plugin>/tmp/），
            # 再由 engine 拷贝进知识库目录（避免直接写 docs 目录后又被
            # engine 重复拷贝而触发文件占用错误）。
            stage_dir = Path(tempfile.mkdtemp(prefix="rag_upload_", dir=str(self.adapter.tmp_dir)))
            saved = []
            try:
                for _name, upload in files.items():
                    if not upload.filename:
                        continue
                    if Path(upload.filename).suffix.lower() not in _SUPPORTED_EXTS:
                        continue
                    dest = stage_dir / Path(upload.filename).name
                    await upload.save(dest)
                    saved.append(dest)
                if not saved:
                    return error_response("没有可导入的支持文件", status_code=400)
                result = await self.adapter.ingest(
                    kb, [str(p) for p in saved]
                )
                return json_response(
                    {"saved": [p.name for p in saved], "index": result}
                )
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)
        except Exception as exc:
            logger.exception("[RAG] web/ingest 失败")
            return error_response(str(exc))

    async def web_rebuild(self):
        try:
            kb = request.query.get("kb_id", self.adapter.default_kb)
            return json_response(await self.adapter.rebuild(kb))
        except Exception as exc:
            logger.exception("[RAG] web/rebuild 失败")
            return error_response(str(exc))

    async def web_progress(self):
        try:
            kb = request.query.get("kb_id", self.adapter.default_kb)
            return json_response(await self.adapter.get_build_progress_dict(kb))
        except Exception as exc:
            logger.exception("[RAG] web/progress 失败")
            return error_response(str(exc))

    # ------------------------------------------------------------------
    # 配置管理
    # ------------------------------------------------------------------

    async def web_config_get(self):
        try:
            return json_response(await self.adapter.get_config_payload())
        except Exception as exc:
            logger.exception("[RAG] web/config 读取失败")
            return error_response(str(exc))

    async def web_config_update(self):
        try:
            body = await request.json(default={})
            return json_response(await self.adapter.update_config(body or {}))
        except Exception as exc:
            logger.exception("[RAG] web/config 保存失败")
            return error_response(str(exc))

    # ------------------------------------------------------------------
    # 知识库 / 文档管理
    # ------------------------------------------------------------------

    async def web_kb_delete(self):
        try:
            body = await request.json(default={})
            kb = (body or {}).get("kb_id") or self.adapter.default_kb
            await self.adapter.delete_kb(kb)
            return json_response({"deleted": kb})
        except Exception as exc:
            logger.exception("[RAG] web/kb/delete 失败")
            return error_response(str(exc))

    async def web_kb_create(self):
        try:
            body = await request.json(default={})
            kb = (body or {}).get("kb_id", "").strip()
            result = await self.adapter.create_kb(kb)
            return json_response({"created": kb, "index": result})
        except Exception as exc:
            logger.exception("[RAG] web/kb/create 失败")
            return error_response(str(exc))

    async def web_kb_select(self):
        try:
            body = await request.json(default={})
            kb = self.adapter.select_kb((body or {}).get("kb_id", ""))
            return json_response({"selected": kb})
        except Exception as exc:
            logger.exception("[RAG] web/kb/select 失败")
            return error_response(str(exc))

    async def web_kb_index_delete(self):
        try:
            body = await request.json(default={})
            kb = (body or {}).get("kb_id") or self.adapter.default_kb
            await self.adapter.drop_index(kb)
            return json_response({"index_deleted": kb})
        except Exception as exc:
            logger.exception("[RAG] web/kb/index/delete 失败")
            return error_response(str(exc))

    async def web_kb_defaults_get(self):
        try:
            return json_response({"kb_ids": self.adapter.get_default_search_kbs()})
        except Exception as exc:
            logger.exception("[RAG] web/kb/defaults 读取失败")
            return error_response(str(exc))

    async def web_kb_defaults_set(self):
        try:
            body = await request.json(default={})
            ids = self.adapter.set_default_search_kbs((body or {}).get("kb_ids", []))
            return json_response({"kb_ids": ids})
        except Exception as exc:
            logger.exception("[RAG] web/kb/defaults 保存失败")
            return error_response(str(exc))

    async def web_kb_documents(self):
        try:
            kb = request.query.get("kb_id", self.adapter.default_kb)
            return json_response(
                {"kb_id": kb, "documents": await self.adapter.list_documents(kb)}
            )
        except Exception as exc:
            logger.exception("[RAG] web/kb/documents 失败")
            return error_response(str(exc))

    async def web_kb_document_delete(self):
        try:
            body = await request.json(default={})
            kb = (body or {}).get("kb_id") or self.adapter.default_kb
            filename = (body or {}).get("filename", "")
            if not filename:
                return error_response("缺少 filename", status_code=400)
            result = await self.adapter.delete_document(kb, filename)
            return json_response({"deleted": filename, "index": result})
        except Exception as exc:
            logger.exception("[RAG] web/kb/document/delete 失败")
            return error_response(str(exc))

    async def terminate(self) -> None:
        await self.adapter.terminate()
