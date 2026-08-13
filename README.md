# astrbot_plugin_RAG_Konwledge_Pro

独立于 AstrBot 原生知识库的增强 RAG 插件：

- **Qdrant** 向量检索（named vector：`text` / `image`）
- **在线 Embedding API**（OpenAI 兼容，支持文本与图片）
- **在线 Reranker API**（SiliconFlow vLLM Rerank）
- 索引版本化 + **自动重建** + 失败回滚（旧索引始终可用）
- **增量索引**：新增 / 修改 / 删除文档按内容 hash 精确同步
- **Embedding 缓存**：避免重建与重复导入时重复调用在线 API
- 文本 / 图片检索结果统一为 `SearchResult`
- WebUI 页面：状态、上传、检索、重建

## 架构

```text
AstrBot
  └─ main.py ── AstrBotRAGAdapter ── RAGEngine
                                       ├─ DocumentParser (txt/md/pdf/image)
                                       ├─ TextChunker
                                       ├─ EmbeddingProvider (OpenAI-compatible)
                                       ├─ EmbeddingCache (SQLite)
                                       ├─ IndexManager (version + rebuild + rollback)
                                       ├─ QdrantStore (VectorStore interface)
                                       ├─ Retriever (Top-K → Rerank → Top-N)
                                       └─ RerankerProvider (SiliconFlow)
```

核心 RAG 逻辑在 `core/`，完全独立于 AstrBot；AstrBot 相关适配集中在 `adapter/` 与 `main.py`。

## 安装 / 依赖

```bash
pip install -r requirements.txt   # qdrant-client, httpx, PyMuPDF, Pillow
```

## 配置

在 AstrBot 插件管理页（或 `_conf_schema.json`）中填写：

- `qdrant_url` / `qdrant_api_key`（本地开发模式可不填 Key）
- `qdrant_collection_prefix`（可选）：Qdrant collection 命名空间前缀。多个 client 共享同一 Qdrant 后端时，每个 client 设为唯一值即可隔离集合（避免误删/误认）；**留空则自动使用本机设备指纹生成的命名空间**（首次生成后持久化到 plugin_data，长期稳定）。修改后新索引使用新前缀，旧索引按 manifest 继续可用
- `embedding_api_base` / `embedding_api_key` / `embedding_model` / `embedding_dimension`
- `rerank_api_base` / `rerank_api_key` / `rerank_model`
- 分块与检索参数：`chunk_size` / `chunk_overlap` / `top_k` / `top_n`

> `embedding_input_type` 用于 NVIDIA 这类非对称模型，默认 `passage`。
> 修改任何影响索引的配置（Embedding 模型 / 维度 / 分块参数）后，
> 插件会自动检测 `config_hash` 变化并触发后台重建，构建成功后才原子切换活动版本。

## 命令

| 命令 | 说明 |
|---|---|
| `/rag status` | 当前索引状态（版本、文档/块数、构建进度） |
| `/rag list` | 知识库列表 |
| `/rag search <问题>` | 检索并返回带来源标注的内容 |
| `/rag rebuild` | 强制重建索引 |
| `/rag ingest` | 导入消息中的 txt/md/pdf/图片文件 |

## LLM 工具

插件注册了 `rag_search` 函数调用工具，模型在回答相关问题时可按需检索知识库，
返回带 `[Source: <文件>, Page: N]` 标注的上下文。

## WebUI

插件页面位于 `pages/rag/`（在 AstrBot WebUI 中打开插件的「RAG」页面）：

- 知识库状态 / 重建进度
- 文档上传（多选，支持 txt/md/pdf/图片）
- 检索测试
- 触发重建

## 数据位置

所有运行时文件（知识库文档、索引元数据、Embedding 缓存、上传临时文件）统一存放于 AstrBot 的
`data/plugin_data/astrbot_plugin_RAG_Konwledge_Pro/`（通过 `StarTools.get_data_dir` 获取），
与插件本体目录完全隔离：

```text
data/plugin_data/astrbot_plugin_RAG_Konwledge_Pro/
├── <kb_id>/            # 知识库：manifests、documents.json、documents/ 文档副本
├── tmp/                # WebUI 上传暂存（用完即清理）
└── cache.db            # Embedding 缓存
```

向量数据在 Qdrant collection `astrbot_rag_<kb_id>_v<version>`。
`e2e_config.json`（含 API Key，gitignore）与 `scripts/e2e_test.py` 为本地开发工具；
e2e 测试数据同样写入 `data/plugin_data/astrbot_plugin_RAG_Konwledge_Pro/e2e/`。

## 测试

```bash
pytest                          # 单元测试（mock，不触网）
python scripts/e2e_test.py      # 端到端（真实 Qdrant + Embedding + Rerank，需 e2e_config.json）
```
