# sqlite-vec-haystack

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Haystack 2.x **DocumentStore + Retriever** backed by [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — embedded vector search in a single SQLite file, cross-platform (Linux / macOS / Windows / Android / iOS / WASM).

## Why this exists

`haystack-core-integrations` covers server-backed vector stores (pgvector, Qdrant, Chroma, Elasticsearch, MongoDB Atlas, ...) but lacks an **embedded, zero-server, single-file** vector store for edge/mobile/on-device RAG. `sqlite-vec` is the modern successor to the deprecated `sqlite-vss` and runs anywhere SQLite runs.

LangChain already has [`SQLiteVec`](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.sqlitevec.SQLiteVec.html); Haystack is the major missing Python RAG framework.

## Installation

```bash
pip install sqlite-vec-haystack
```

## Usage

```python
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore
from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever

# 1. Set up the store
store = SQLiteVecDocumentStore(
    db_path="corpus.db",
    embedding_dim=384,            # must match your embedder
    distance_metric="cosine",     # "cosine" | "l2" | "dot"
)

# 2. Embed and write documents (embedding is composed externally — same as pgvector / chroma)
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

## Status

Early-stage (0.1.x). Implemented in this release:

- `write_documents` with `OVERWRITE` / `SKIP` / `NONE` / `FAIL` policies
- `count_documents`, `delete_documents`
- KNN retrieval via `SQLiteVecEmbeddingRetriever`
- `to_dict` / `from_dict` for pipeline serialization

Planned for upcoming releases:

- Metadata filtering (Haystack filter DSL → SQL `json_extract` predicates)
- Async API (`*_async` methods)
- Full mixin test suite

## License

[Apache-2.0](LICENSE)
