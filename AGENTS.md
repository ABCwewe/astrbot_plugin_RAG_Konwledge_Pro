# AGENTS.md

# AstrBot RAG Plugin

## 1. 项目目标

本项目是一个 AstrBot 插件，用于提供独立于 AstrBot 原生知识库检索实现的增强型 RAG（Retrieval-Augmented Generation）能力。

插件必须具备以下核心能力：

1. 使用现成的向量数据库作为 RAG 后端。
2. 支持自定义文本分块规则。
3. 支持在线 Embedding API。
4. 支持在线 Reranker API。
5. 支持文本向量检索。
6. 支持图片向量检索。
7. 支持文本和图片结果的统一检索结果模型。
8. 更换 Embedding 模型后自动检测索引配置变化。
9. Embedding 模型变化后自动创建新索引并重建向量。
10. 重建过程中不影响当前可用的旧索引。
11. 新索引构建成功后原子切换 active index。
12. 支持索引失败后的安全回滚。
13. 支持增量添加、修改、删除文档。
14. 支持 Embedding 缓存，避免重复调用在线模型 API。
15. 尽可能保持轻量、快速、易于维护。
16. 尽量避免引入大型 RAG Framework。

---

# 2. 当前运行环境

开发和测试目标环境：

- AstrBot 4.27.1
- Python 3.12
- uv / virtualenv
- Linux
- 异步 Python 环境

代码必须兼容 Python 3.12。

优先使用 Python 标准库以及体积较小、职责明确的第三方依赖。

---

# 3. 核心技术选型

## 3.1 向量数据库

必须使用 Qdrant 作为第一版默认 Vector Store。

推荐通过：

```text
qdrant-client
```

访问。

不得自行实现：

- HNSW
- ANN
- 向量索引
- 向量数据库

Qdrant 只负责：

- 向量保存
- 向量搜索
- metadata/payload
- filter
- collection 管理

RAG 业务逻辑必须位于插件自身。

---

# 4. 总体架构

系统架构：

```text
AstrBot
    |
    v
RAG Plugin
    |
    +-- Document Parser
    |
    +-- Chunker
    |
    +-- Embedding Provider
    |
    +-- Index Manager
    |
    +-- Qdrant Store
    |
    +-- Retriever
    |
    +-- Reranker Provider
    |
    +-- AstrBot Adapter
    |
    v
RAG Context
    |
    v
AstrBot LLM
```

核心模块：

```text
core/
    engine.py
    models.py
    chunking/
    parsers/
    providers/
    storage/
    indexing/
    retrieval/
```

---

# 5. 架构原则

## 5.1 业务逻辑与 Vector Store 解耦

业务代码不得直接依赖 Qdrant API。

错误：

```python
client.search(...)
```

出现在 Retriever 或 Engine 中。

正确：

```text
Retriever
    |
    v
VectorStore interface
    |
    v
QdrantStore
```

以后如果更换 Vector Store，只允许修改 storage 层。

---

## 5.2 Embedding 与 RAG 解耦

RAGEngine 不得直接调用某个具体 Embedding API。

必须通过：

```python
EmbeddingProvider
```

访问。

第一版实现：

```text
OpenAICompatibleEmbedding
```

---

## 5.3 Reranker 与 RAG 解耦

RAGEngine 不得直接调用具体 Reranker HTTP API。

必须通过：

```python
RerankerProvider
```

访问。

---

## 5.4 AstrBot API 与核心 RAG 解耦

AstrBot API 适配代码必须集中在：

```text
AstrBotRAGAdapter
main.py
```

核心 RAG Engine 不得直接依赖 AstrBot 内部对象。

---

# 6. 数据模型

必须至少存在以下业务模型。

## Document

```python
@dataclass
class Document:
    id: str
    source: str
    filename: str
    content_hash: str
    metadata: dict
```

## Chunk

```python
@dataclass
class Chunk:
    id: str
    document_id: str
    type: Literal["text", "image"]
    content: str | None
    image_path: str | None
    chunk_index: int
    metadata: dict
```

## SearchResult

```python
@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str | None
    image_path: str | None
    vector_score: float
    rerank_score: float | None
    metadata: dict
```

不得让 Qdrant Point 直接充当业务对象。

---

# 7. Chunking

第一版必须支持：

```yaml
chunking:
  separator: "\n\n"
  chunk_size: 800
  chunk_overlap: 100
```

参数：

- separator：自定义分块符
- chunk_size：最大字符数量
- chunk_overlap：重叠字符数量

第一版 chunk_size 使用字符数量，不绑定任何 tokenizer。

不要引入复杂的 LangChain splitter。

Chunker 接口：

```python
class TextChunker:

    def split(self, text: str) -> list[str]:
        ...
```

必须正确处理：

- 空文本
- 连续 separator
- 超长文本
- overlap
- separator 不存在
- 最后一块不足 chunk_size

---

# 8. Document Parser

必须采用 Parser 抽象。

接口：

```python
class DocumentParser(ABC):

    @abstractmethod
    def supports(self, path: Path) -> bool:
        ...

    @abstractmethod
    async def parse(self, path: Path) -> ParsedDocument:
        ...
```

第一版至少支持：

- TXT
- Markdown
- PDF
- 图片

PDF 必须保留 page metadata。

不得把整个 PDF 简单合并为一段无页码文本。

---

# 9. 图片

图片必须作为独立 Chunk 类型：

```text
type = "image"
```

图片 Chunk 至少保存：

- image_path
- document_id
- source
- page（如果存在）
- chunk_id

第一版图片检索重点是 image embedding。

OCR 不属于第一版强制需求。

不要为了图片检索强制依赖 OCR。

---

# 10. Embedding Provider

必须提供抽象接口：

```python
class EmbeddingProvider(ABC):

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    @property
    def supports_text(self) -> bool:
        ...

    @property
    def supports_image(self) -> bool:
        ...

    async def embed_text(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...

    async def embed_image(
        self,
        images: list[bytes],
    ) -> list[list[float]]:
        ...
```

文本 Embedding 和图片 Embedding 必须允许分别实现。

不得假设所有 Embedding API 都支持图片。

---

# 11. Online API

第一版默认使用 OpenAI-compatible API。

HTTP 客户端必须优先使用：

```text
httpx.AsyncClient
```

不得在异步业务路径中使用同步 requests。

Embedding 必须支持批量请求。

禁止：

```python
for chunk in chunks:
    await embed(chunk)
```

优先：

```text
chunks
    |
    v
batch
    |
    v
Embedding API
```

必须提供并发限制。

推荐默认：

```text
embedding concurrency = 4
```

具体数值必须可配置。

---

# 12. Reranker

接口：

```python
class RerankerProvider(ABC):

    @property
    def model_name(self) -> str:
        ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        ...
```

典型检索流程：

```text
Embedding
    |
    v
Qdrant Top-K
    |
    v
Reranker
    |
    v
Top-N
```

默认推荐：

```text
top_k = 30
top_n = 6
```

参数必须可配置。

---

# 13. Qdrant

一个知识库使用一个逻辑 collection。

Collection 名称必须包含版本：

```text
astrbot_rag_<kb_id>_v<version>
```

例如：

```text
astrbot_rag_default_v1
astrbot_rag_default_v2
```

禁止使用无版本 collection 作为主要索引。

---

# 14. Named Vector

如果同时存在文本和图片向量，应使用不同的 named vector。

例如：

```text
text
image
```

不得强制要求图片向量和文本向量维度一致。

---

# 15. Payload

Qdrant payload 至少应该包含：

```json
{
  "chunk_id": "...",
  "document_id": "...",
  "type": "text",
  "content": "...",
  "source": "...",
  "page": 3,
  "chunk_index": 5
}
```

图片：

```json
{
  "chunk_id": "...",
  "document_id": "...",
  "type": "image",
  "image_path": "...",
  "source": "...",
  "page": 3
}
```

payload 必须足以恢复 SearchResult。

---

# 16. Index Manifest

每个知识库必须存在 Manifest。

Manifest 至少记录：

```text
schema_version
kb_id
version
status
embedding provider
embedding model
embedding dimension
image embedding provider
image embedding model
image embedding dimension
chunking configuration
document count
chunk count
collection name
config_hash
```

Manifest 是判断是否需要重建索引的唯一依据之一。

---

# 17. Config Hash

必须计算影响索引的配置 hash。

至少包括：

```text
embedding provider
embedding model
embedding dimension
image embedding provider
image embedding model
image embedding dimension
chunk separator
chunk size
chunk overlap
```

如果这些参数发生变化：

```text
config_hash changed
```

必须触发索引重建。

---

# 18. 自动重建

绝对禁止：

```text
删除旧 collection
↓
重新建立
```

正确流程：

```text
active v2
    |
    v
创建 v3
    |
    v
BUILDING
    |
    v
重新解析文档
    |
    v
重新 Chunk
    |
    v
Embedding
    |
    v
写入 v3
    |
    v
完整性验证
    |
    v
READY
    |
    v
active = v3
    |
    v
删除 v2
```

重建失败：

```text
v3 = FAILED
active = v2
```

旧索引必须保持可用。

---

# 19. 索引状态

至少支持：

```text
CREATING
BUILDING
READY
FAILED
DELETING
```

状态必须持久化。

---

# 20. 增量索引

新增文档：

```text
hash
↓
不存在
↓
index
```

修改文档：

```text
hash changed
↓
delete old chunks
↓
re-index
```

删除文档：

```text
delete document
↓
delete all chunks with document_id
```

不得每次增加一个文件都完整重建整个知识库。

---

# 21. Document Hash

必须使用 hash 判断文件是否变化。

推荐：

```text
SHA-256
```

Hash 至少基于原始文件内容。

---

# 22. Embedding Cache

必须支持 Embedding cache。

Cache key 至少包含：

```text
embedding model
content hash
```

图片：

```text
image hash
image embedding model
```

缓存命中时不得重新请求远程 Embedding API。

---

# 23. Cache 目的

主要用于：

1. 自动重建。
2. 模型切换后的部分重复内容。
3. 文档重复导入。
4. API 失败后的恢复。
5. 降低在线 API 成本。

---

# 24. Retrieval

标准流程：

```text
query
    |
    v
query embedding
    |
    v
Qdrant Top-K
    |
    v
candidate chunks
    |
    v
Reranker
    |
    v
Top-N
```

Retrieval 不得直接依赖具体 HTTP API。

---

# 25. 图片检索

默认普通文本查询只进行文本向量检索。

图片向量检索必须可配置。

原因：

- 降低 API 调用次数。
- 降低响应延迟。
- 避免普通文本查询无意义地调用图片 embedding API。

配置可包含：

```yaml
image:
  enabled: true
  search_always: false
```

当需要图片检索时：

```text
text search
+
image search
↓
merge
↓
rerank
```

---

# 26. 图片 Rerank

普通文本 Reranker 通常不能直接理解图片。

因此图片结果进入普通 Reranker 前，应有可用的文本表示，例如：

```text
caption
```

或者：

```text
source + page + metadata
```

未来支持 multimodal reranker 时，再扩展：

```python
supports_image
```

第一版不强制实现 multimodal reranker。

---

# 27. 并发

必须限制远程 API 并发。

Embedding：

```text
Semaphore
```

Reranker：

```text
Semaphore
```

同一个知识库：

```text
最多一个 rebuild task
```

多个 rebuild 请求不得同时执行。

---

# 28. 异步

网络操作必须异步。

优先：

```text
asyncio
httpx.AsyncClient
qdrant async client
```

CPU 密集或同步第三方库必须使用：

```python
asyncio.to_thread(...)
```

不得阻塞 AstrBot event loop。

---

# 29. 错误处理

必须区分：

```text
EmbeddingAPIError
RerankerAPIError
QdrantError
ParserError
IndexBuildError
ConfigurationError
```

远程 API 失败时：

```text
新 index = FAILED
旧 index = READY
```

不得因为一次重建失败导致整个知识库不可用。

---

# 30. Retry

远程 API 可以进行有限次数 retry。

推荐：

```text
max retries = 3
```

使用指数退避。

不得无限重试。

4xx 配置错误通常不应无限 retry。

---

# 31. AstrBot 集成

`main.py` 只负责：

- 插件生命周期
- AstrBot API 注册
- Command
- Event Hook
- Adapter

不得把：

- Chunking
- Embedding
- Qdrant
- Indexing

直接写入 `main.py`。

---

# 32. AstrBot Adapter

必须存在适配层：

```text
AstrBot
    |
    v
AstrBotRAGAdapter
    |
    v
RAGEngine
```

AstrBot API 发生变化时，优先修改 Adapter。

---

# 33. 管理命令

第一版至少提供：

```text
/rag list
/rag status
/rag search <query>
/rag rebuild
```

这些命令主要用于调试和管理员使用。

---

# 34. Plugin Page

应提供基本 WebUI。

至少包含：

```text
Knowledge Bases
Embedding Configuration
Reranker Configuration
Chunking Configuration
Retrieval Configuration
Index Status
Rebuild Button
```

重建时显示：

```text
status
current version
target version
processed documents
processed chunks
progress
error
```

---

# 35. 日志

日志应包含模块前缀：

```text
[RAG]
[INDEX]
[EMBED]
[QDRANT]
[RERANK]
```

例如：

```text
[RAG][INDEX] Building default v3
[RAG][EMBED] Processing batch 12/40
[RAG][QDRANT] Upserted 320 points
[RAG][INDEX] Build completed
[RAG][INDEX] Switched active version to v3
```

禁止输出：

- API key
- Authorization header
- 用户隐私数据
- 完整远程请求内容

---

# 36. 配置安全

API Key 不得：

- 写入日志
- 写入异常信息
- 返回给 WebUI 前端
- 保存到普通 debug dump

前端只显示：

```text
********
```

---

# 37. 性能要求

目标是轻量快速。

禁止默认引入：

- LangChain
- LlamaIndex
- Haystack
- 大型 Agent Framework

除非未来明确证明某个功能无法合理实现。

第一版核心依赖尽量控制在：

```text
qdrant-client
httpx
PyMuPDF
Pillow
```

以及 AstrBot 自身依赖。

---

# 38. 依赖原则

新依赖必须满足至少一个条件：

1. 显著减少代码量。
2. 提供成熟且难以自行实现的核心功能。
3. 明显提升稳定性。

不能因为“方便”就增加大型 Framework。

---

# 39. 测试

必须为核心纯逻辑提供单元测试。

至少：

```text
test_chunker.py
test_manifest.py
test_provider.py
test_indexer.py
test_retriever.py
```

重点测试：

- chunking
- hash
- config change detection
- versioning
- rebuild
- rollback
- incremental indexing
- reranking
- empty result
- API failure

---

# 40. Mock

单元测试不得依赖真实：

- Qdrant Cloud
- Embedding API
- Reranker API

必须使用 mock/fake provider。

真实服务只能用于 integration test。

---

# 41. Rebuild 安全性

这是项目最高优先级之一。

任何情况下：

```text
新索引未 READY
```

都不能：

```text
active = new index
```

任何情况下：

```text
new index FAILED
```

都必须：

```text
active = old index
```

---

# 42. 数据一致性

切换索引前至少检查：

```text
collection exists
vector count > 0（对于非空知识库）
document count
chunk count
manifest
```

确认一致后才能切换 active version。

---

# 43. ID 设计

Document ID 必须稳定。

推荐：

```text
SHA-256(source identity)
```

Chunk ID：

```text
document_id + chunk_index
```

图片：

```text
document_id + image identifier
```

不要使用随机 UUID 作为唯一业务 ID，否则增量更新无法稳定判断。

---

# 44. Source Identity

Document ID 不应该只根据文件内容生成。

推荐：

```text
knowledge_base_id
+
source path / source identifier
```

生成稳定 document_id。

然后：

```text
content_hash
```

单独判断文件是否变化。

---

# 45. RAG Context

SearchResult 最终必须能够转换成：

```text
source
page
content
metadata
```

例如：

```text
[Source: manual.pdf, Page: 3]

这里是相关内容……
```

最终 Context 格式不要和 Qdrant payload 格式绑定。

---

# 46. 第一版明确不实现

以下功能不属于 MVP：

```text
Graph RAG
Agentic RAG
HyDE
Query Rewrite
BM25
Sparse Vector
复杂 Hybrid Search
ColBERT
本地 embedding 推理
本地 reranker 推理
自动 OCR
自动摘要
复杂 Query Planner
```

架构必须为未来扩展留下 Provider/Strategy 接口，但当前不得为了未来功能过度设计。

---

# 47. 推荐开发阶段

## Phase 1

实现：

```text
models
exceptions
config
chunker
```

目标：

```text
文本 → chunks
```

---

## Phase 2

实现：

```text
EmbeddingProvider
OpenAICompatibleEmbedding
QdrantStore
```

目标：

```text
文本
↓
embedding
↓
Qdrant
↓
search
```

---

## Phase 3

实现：

```text
DocumentParser
Indexer
Manifest
IndexManager
```

目标：

```text
文档导入
+
增量更新
+
自动重建
+
版本切换
```

---

## Phase 4

实现：

```text
RerankerProvider
Retriever
```

目标：

```text
Top-K
↓
Rerank
↓
Top-N
```

---

## Phase 5

实现：

```text
AstrBotRAGAdapter
main.py
commands
```

目标：

```text
AstrBot
↓
Plugin
↓
RAG
↓
LLM
```

---

## Phase 6

实现：

```text
PDF
Image
Image Embedding
```

目标：

```text
text retrieval
+
image retrieval
```

---

## Phase 7

实现：

```text
Plugin Page
```

目标：

```text
可视化配置
+
索引状态
+
重建
```

---

# 48. 开发顺序约束

开发 Agent 应严格遵循：

```text
models
↓
exceptions
↓
config
↓
chunker
↓
providers
↓
storage
↓
manifest
↓
indexer
↓
retriever
↓
engine
↓
AstrBot adapter
↓
main
↓
UI
```

不得在核心模块尚未稳定时优先开发 UI。

---

# 49. 代码质量要求

要求：

- 类型标注
- async/await
- 小函数
- 单一职责
- 清晰异常
- 不重复代码
- 不隐藏网络请求
- 不在业务层直接操作 Qdrant
- 不在 main.py 实现业务逻辑

优先可读性，不追求过度抽象。

---

# 50. 最重要的工程原则

本项目的核心不是“做一个很复杂的 RAG Framework”。

目标是：

```text
简单
+
稳定
+
快速
+
可替换模型
+
自动重建
+
低维护成本
```

如果一个功能可以通过 100 行清晰 Python 实现，不应为了“通用性”引入一个大型 Framework。

如果一个功能会让核心架构复杂数倍，应推迟到后续版本。

---

# 51. 完成标准

当以下流程完整工作时，MVP 才算完成：

```text
上传 Markdown
      ↓
Chunk
      ↓
Online Embedding
      ↓
Qdrant
      ↓
用户提问
      ↓
Embedding
      ↓
Top-K
      ↓
Reranker
      ↓
Top-N
      ↓
AstrBot LLM
```

并且：

```text
修改 Embedding Model
      ↓
检测 config hash
      ↓
创建新 collection
      ↓
后台重建
      ↓
新 collection READY
      ↓
active version 切换
      ↓
旧 collection 删除
```

重建失败时：

```text
旧 collection
仍然可用
```

这两条链路是整个项目的核心验收标准。