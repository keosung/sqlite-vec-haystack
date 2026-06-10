# sqlite-vec-haystack

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A [Haystack](https://haystack.deepset.ai/) 2.x **DocumentStore + Retriever** backed by [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — embedded vector search in a single SQLite file, cross-platform (Linux / macOS / Windows / Android / iOS / WASM).

[Haystack](https://github.com/deepset-ai/haystack) is deepset's open-source framework for building LLM applications and RAG pipelines out of composable components. [sqlite-vec](https://github.com/asg017/sqlite-vec) is a pure-C SQLite extension that adds vector search to any SQLite database. This package connects the two.

## Why this exists

Most Haystack document stores are backed by a server process. This one is a single SQLite file: nothing to run, nothing to configure, and the indexed corpus can be copied to another machine or device as-is. That makes it a good fit for local-first applications and on-device RAG. `sqlite-vec` is the successor to the deprecated `sqlite-vss` and runs anywhere SQLite runs.

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

Retrieved documents carry the raw distance reported by sqlite-vec in `Document.score`, ordered ascending: **lower score means closer**. For `cosine`, `0.0` is an identical direction. The score is `None` when the metric is undefined for a pair (for example cosine against a zero vector).

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

The retriever also supports `filter_policy` (`"replace"` by default, or `"merge"`) to control how init-time and runtime filters combine, matching the convention used by other Haystack retrievers.

## Async API

Every store method has an `*_async` counterpart, and the retriever exposes `run_async` so it drops into `AsyncPipeline` without further wiring:

```python
import asyncio
from haystack import AsyncPipeline

retriever = SQLiteVecEmbeddingRetriever(document_store=store, top_k=5)
pipeline = AsyncPipeline()
pipeline.add_component("retriever", retriever)

async def main() -> None:
    # any Haystack text embedder produces the query embedding
    query_embedding = text_embedder.run(text="What is the capital of Germany?")["embedding"]
    result = await pipeline.run_async({"retriever": {"query_embedding": query_embedding}})
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

See [CHANGELOG.md](CHANGELOG.md) for the release history.

## Limitations

- **Documents must carry an embedding when written.** Embedding-less writes
  (the pattern Haystack's shared test mixins assume) are planned but not
  supported yet; today `write_documents` raises `ValueError` for them.
- **Search is exact brute-force KNN.** sqlite-vec has no ANN index, so every
  query scans all vectors. This is exact (perfect recall) and fast for
  on-device corpus sizes — about 30 ms for 30K × 768-dim vectors — but it is
  not the right backend for millions of documents.
- **`filter_documents` returns Documents without embeddings.** Embeddings
  live in the vec0 table and are only used for retrieval.
- **On-device targets need a sqlite-vec build for that platform.** The
  `sqlite-vec` PyPI wheels cover Linux / macOS / Windows; for Android, iOS,
  or WASM you load the same `.db` file, but the vec0 extension itself must be
  compiled for the target.
- **One process at a time for writes.** A single store instance is fully
  thread-safe (internal lock), and SQLite's WAL mode handles concurrent
  readers, but multiple writer processes are serialized by SQLite's own file
  locking with no further coordination from this package.

## Development

```bash
git clone https://github.com/keosung/sqlite-vec-haystack.git
cd sqlite-vec-haystack
pip install -e .
pip install pytest pytest-asyncio ruff

pytest                      # run the test suite
ruff check src tests        # lint
ruff format --check src tests
```

[Hatch](https://hatch.pypa.io/) users can run the same things via the
configured environments: `hatch run test:all`, `hatch run fmt-check`.

Bug reports and pull requests are welcome. If you are proposing a behaviour
change, please include a test that demonstrates it.

## License

[Apache-2.0](LICENSE)
