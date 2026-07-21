# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document

try:  # Haystack 2.x ships a dedicated AsyncPipeline
    from haystack import AsyncPipeline
except ImportError:  # Haystack 3.x folds run_async into Pipeline
    from haystack import Pipeline as AsyncPipeline

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore

pytestmark = pytest.mark.asyncio


def _doc(id_: str, *, meta: dict | None = None, embedding: list[float] | None = None) -> Document:
    return Document(
        id=id_,
        content=f"content of {id_}",
        meta=meta or {},
        embedding=embedding or [0.1, 0.2, 0.3, 0.4],
    )


class TestRunAsync:
    async def test_basic(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [
                _doc("near", embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc("far", embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=2)
        out = await retriever.run_async(query_embedding=[1.0, 0.0, 0.0, 0.0])
        assert [d.id for d in out["documents"]] == ["near", "far"]

    async def test_with_runtime_filter(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [
                _doc("a", meta={"lang": "en"}, embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc("b", meta={"lang": "ko"}, embedding=[1.0, 0.0, 0.0, 0.0]),
            ]
        )
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=5)
        out = await retriever.run_async(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
        )
        assert [d.id for d in out["documents"]] == ["b"]

    async def test_haystack_supports_async_flag(self, document_store: SQLiteVecDocumentStore):
        """The @component decorator sets this flag based on run_async presence."""
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store)
        assert getattr(retriever, "__haystack_supports_async__", False) is True

    async def test_runs_inside_async_pipeline(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [
                _doc("a", embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc("b", embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=2)

        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)
        result = await pipeline.run_async({"retriever": {"query_embedding": [1.0, 0.0, 0.0, 0.0]}})
        assert "retriever" in result
        assert len(result["retriever"]["documents"]) == 2
