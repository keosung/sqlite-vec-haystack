# sqlite-vec-haystack

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Haystack 2.x **DocumentStore + Retriever** backed by [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — embedded vector search in a single SQLite file, cross-platform (Linux / macOS / Windows / Android / iOS / WASM).

## Why this exists

`haystack-core-integrations` covers server-backed vector stores (pgvector, Qdrant, Chroma, Elasticsearch, MongoDB Atlas, …) but none of them target the **embedded, zero-server, single-file** case for edge or on-device RAG. `sqlite-vec` is the modern successor to the deprecated `sqlite-vss` and runs anywhere SQLite runs.

LangChain already ships [`SQLiteVec`](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.sqlitevec.SQLiteVec.html); Haystack is the major Python RAG framework still missing this option.

## Installation

```bash
pip install sqlite-vec-haystack
```

This pulls in `haystack-ai>=2.27.0` and `sqlite-vec>=0.1.6` automatically.

## At a glance

```python
from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore

# 1. Set up the store
store = SQLiteVecDocumentStore(
    db_path="corpus.db",
    embedding_dim=384,
    distance_metric="cosine",  # "cosine" | "l2" | "dot"
)

# 2. Index documents (embedding is produced upstream by any Haystack embedder)
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs = doc_embedder.run([Document(content="Berlin is the capital of Germany.")])["documents"]
store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

# 3. Retrieve via a Haystack pipeline
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = SQLiteVecEmbeddingRetriever(document_store=store, top_k=5)

pipeline = Pipeline()
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("retriever", retriever)
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = pipeline.run({"text_embedder": {"text": "What is the capital of Germany?"}})
print(result["retriever"]["documents"][0].content)
```

## Metadata filtering

`SQLiteVecDocumentStore.filter_documents()` and the retriever both accept the standard Haystack filter DSL. Field paths into `meta.*` are translated to `json_extract(meta_json, '$.path')`; values flow through bound parameters so user-supplied field names never reach the SQL string.

```python
# Constrain a vector search to a specific subset
result = retriever.run(
    query_embedding=embedding,
    filters={"field": "meta.lang", "operator": "==", "value": "ko"},
)

# Nested logic and any operator from the Haystack DSL works
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.year", "operator": ">=", "value": 2024},
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "biology"},
                {"field": "meta.category", "operator": "==", "value": "chemistry"},
            ],
        },
    ],
}
```

Supported operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`, plus `AND` / `OR` / `NOT`. The filter is pushed into a `rowid IN (...)` subquery so vec0 only scores matching candidates — selective filters stay fast and `top_k` is always exact.

## Async API

Every store method has an `*_async` counterpart, and the retriever exposes `run_async` so it drops into `AsyncPipeline` without further wiring:

```python
import asyncio
from haystack import AsyncPipeline

retriever = SQLiteVecEmbeddingRetriever(document_store=store, top_k=5)
pipeline = AsyncPipeline()
pipeline.add_component("retriever", retriever)

async def main() -> None:
    result = await pipeline.run_async({"retriever": {"query_embedding": embedding}})
    print(result["retriever"]["documents"][0].content)

asyncio.run(main())
```

Under the hood the store dispatches each `*_async` call through `asyncio.to_thread` and serialises DB access with a `threading.RLock`. Concurrent `asyncio.gather` over the same store is safe; benchmarks (`scripts/bench_lock_overhead.py`) show the lock adds no measurable throughput cost.

## Embedded / on-device RAG

The store is a single SQLite file. Index on a host, then ship the file to any platform that has SQLite and the sqlite-vec extension:

```bash
# Host: build corpus.db
python examples/embedding_retrieval.py

# Device (Android, iOS, Pi, browser via WASM, …)
adb push corpus.db /sdcard/
```

The `.db` is fully portable across platforms — endianness, alignment and the on-disk format are all owned by SQLite.

## Examples

Runnable scripts in `examples/`:

- `quickstart.py` — write / retrieve / filter with synthetic embeddings (no model download)
- `async_pipeline.py` — `AsyncPipeline` + concurrent writes via `asyncio.gather`
- `embedding_retrieval.py` — full RAG with `sentence-transformers/all-MiniLM-L6-v2`

## Status

| Capability | Status |
| --- | --- |
| `write_documents` with `OVERWRITE` / `SKIP` / `NONE` / `FAIL` | ✅ |
| `count_documents`, `delete_documents`, `filter_documents` | ✅ |
| KNN retrieval via `SQLiteVecEmbeddingRetriever` | ✅ |
| Metadata filtering on KNN queries | ✅ |
| Async API + thread-safe concurrent access | ✅ |
| `to_dict` / `from_dict` for pipeline serialization | ✅ |
| `cosine` / `l2` / `dot` distance metrics | ✅ |

## License

[Apache-2.0](LICENSE)
