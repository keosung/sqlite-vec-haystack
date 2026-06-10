# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end embedding retrieval with a real sentence-transformers model.

This is the pattern that ``pgvector-haystack`` and ``qdrant-haystack``
examples follow: a full indexing pipeline (embedder → writer) and a
separate query pipeline (text embedder → retriever).

Install the optional dependency first:

    pip install "sentence-transformers>=2.2.0"

Then run:

    python examples/embedding_retrieval.py

A ~25 MB MiniLM model downloads on the first run; the resulting
``corpus.db`` is portable — copy or ``adb push`` it to any device with
the same SQLite + sqlite-vec extension installed.
"""

from __future__ import annotations

from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
DB_PATH = Path("corpus.db")


def build_indexing_pipeline(store: SQLiteVecDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE),
    )
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def build_query_pipeline(store: SQLiteVecDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=MODEL))
    pipeline.add_component("retriever", SQLiteVecEmbeddingRetriever(document_store=store, top_k=3))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return pipeline


def main() -> None:
    raw_docs = [
        Document(content="Berlin is the capital of Germany.", meta={"category": "geography"}),
        Document(content="Paris is the capital of France.", meta={"category": "geography"}),
        Document(content="Seoul is the capital of South Korea.", meta={"category": "geography"}),
        Document(content="Photosynthesis converts sunlight to chemical energy.", meta={"category": "biology"}),
        Document(content="Mitochondria are the powerhouse of the cell.", meta={"category": "biology"}),
    ]

    store = SQLiteVecDocumentStore(
        db_path=DB_PATH,
        embedding_dim=384,  # must match MiniLM-L6
        distance_metric="cosine",
    )

    print("Indexing...")
    indexing = build_indexing_pipeline(store)
    indexing.run({"embedder": {"documents": raw_docs}})
    print(f"  store now holds {store.count_documents()} documents at {DB_PATH}")

    print("\nQuerying...")
    query_pipeline = build_query_pipeline(store)

    # 1. Plain retrieval
    result = query_pipeline.run({"text_embedder": {"text": "Where is the German capital?"}})
    print("Top-3 for 'Where is the German capital?':")
    for d in result["retriever"]["documents"]:
        print(f"  score={d.score:.4f}  '{d.content}'")

    # 2. Same query, filtered to a specific category. Filters can be passed
    #    either at retriever init time or per call.
    biology_only = query_pipeline.run(
        {
            "text_embedder": {"text": "How do cells make energy?"},
            "retriever": {"filters": {"field": "meta.category", "operator": "==", "value": "biology"}},
        }
    )
    print("\nTop-3 for 'How do cells make energy?' filtered to category=='biology':")
    for d in biology_only["retriever"]["documents"]:
        print(f"  score={d.score:.4f}  '{d.content}'")

    store.close()


if __name__ == "__main__":
    main()
