# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Run a SQLiteVec retriever inside a Haystack AsyncPipeline.

This demonstrates the async surface of the integration:

- ``write_documents_async`` / ``filter_documents_async`` on the store
- ``SQLiteVecEmbeddingRetriever.run_async`` dispatched by ``AsyncPipeline``
- thread-safe access from concurrent ``asyncio.gather`` calls

    python examples/async_pipeline.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from haystack import AsyncPipeline, Document

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


async def main() -> None:
    with TemporaryDirectory() as tmp:
        store = SQLiteVecDocumentStore(db_path=Path(tmp) / "async.db", embedding_dim=4)

        # Three batches written concurrently
        batches = [
            [Document(id=f"a{i}", content=f"text a{i}", embedding=[1.0, 0.0, 0.0, 0.0]) for i in range(5)],
            [Document(id=f"b{i}", content=f"text b{i}", embedding=[0.0, 1.0, 0.0, 0.0]) for i in range(5)],
            [Document(id=f"c{i}", content=f"text c{i}", embedding=[0.0, 0.0, 1.0, 0.0]) for i in range(5)],
        ]
        results = await asyncio.gather(*(store.write_documents_async(b) for b in batches))
        print(f"Concurrently wrote {sum(results)} docs across 3 batches.")
        print(f"Final count via async: {await store.count_documents_async()}")

        # Wire the retriever into an AsyncPipeline. The `@component` decorator
        # detects `run_async` and sets `__haystack_supports_async__ = True`,
        # so AsyncPipeline dispatches through that path automatically.
        retriever = SQLiteVecEmbeddingRetriever(document_store=store, top_k=3)
        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)

        # Run two queries concurrently — the store's internal lock serialises
        # actual DB access so the pipeline never observes a half-written batch.
        async def query(label: str, q: list[float]) -> list[Document]:
            out = await pipeline.run_async({"retriever": {"query_embedding": q}})
            print(f"\nQuery {label} top-3:")
            for d in out["retriever"]["documents"]:
                print(f"  {d.id:5s}  score={d.score:.4f}  '{d.content}'")
            return out["retriever"]["documents"]

        await asyncio.gather(
            query("A", [1.0, 0.0, 0.0, 0.0]),
            query("B", [0.0, 0.0, 1.0, 0.0]),
        )

        store.close()


if __name__ == "__main__":
    asyncio.run(main())
