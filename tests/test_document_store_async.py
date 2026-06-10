# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore

pytestmark = pytest.mark.asyncio


def _doc(id_: str, *, meta: dict | None = None, embedding: list[float] | None = None) -> Document:
    return Document(
        id=id_,
        content=f"content of {id_}",
        meta=meta or {},
        embedding=embedding or [0.1, 0.2, 0.3, 0.4],
    )


class TestCountAsync:
    async def test_empty(self, document_store: SQLiteVecDocumentStore):
        assert await document_store.count_documents_async() == 0

    async def test_after_writes(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async([_doc("a"), _doc("b")])
        assert await document_store.count_documents_async() == 2


class TestWriteAsync:
    async def test_returns_number_written(self, document_store: SQLiteVecDocumentStore):
        assert await document_store.write_documents_async([_doc("a"), _doc("b")]) == 2

    async def test_overwrite_idempotent(self, document_store: SQLiteVecDocumentStore):
        docs = [_doc("a"), _doc("b")]
        await document_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)
        await document_store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)
        assert await document_store.count_documents_async() == 2

    async def test_fail_raises_duplicate(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async([_doc("a")], policy=DuplicatePolicy.FAIL)
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async([_doc("a")], policy=DuplicatePolicy.FAIL)


class TestFilterAsync:
    async def test_no_filters(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async(
            [
                _doc("a", meta={"lang": "en"}),
                _doc("b", meta={"lang": "ko"}),
            ]
        )
        result = await document_store.filter_documents_async()
        assert sorted(d.id for d in result) == ["a", "b"]

    async def test_meta_filter(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async(
            [
                _doc("a", meta={"lang": "en"}),
                _doc("b", meta={"lang": "ko"}),
                _doc("c", meta={"lang": "en"}),
            ]
        )
        result = await document_store.filter_documents_async({"field": "meta.lang", "operator": "==", "value": "en"})
        assert sorted(d.id for d in result) == ["a", "c"]


class TestDeleteAsync:
    async def test_delete_subset(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async([_doc("a"), _doc("b"), _doc("c")])
        await document_store.delete_documents_async(["a", "c"])
        result = await document_store.filter_documents_async()
        assert [d.id for d in result] == ["b"]

    async def test_delete_empty_list_is_noop(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async([_doc("a")])
        await document_store.delete_documents_async([])
        assert await document_store.count_documents_async() == 1


class TestConcurrentAsync:
    """`asyncio.gather` on multiple async DB calls must be safe.

    Without locking, the default thread pool dispatches each ``to_thread`` call
    to an arbitrary worker, and concurrent writers race on a shared sqlite
    connection. The internal ``threading.RLock`` serializes everything so the
    end state is deterministic.
    """

    async def test_concurrent_writes_do_not_corrupt(self, document_store: SQLiteVecDocumentStore):
        batches = [[_doc(f"batch{b}-{i}") for i in range(5)] for b in range(8)]
        results = await asyncio.gather(*(document_store.write_documents_async(b) for b in batches))
        assert sum(results) == 40
        assert await document_store.count_documents_async() == 40

    async def test_concurrent_reads_and_writes(self, document_store: SQLiteVecDocumentStore):
        # Mix writes and reads simultaneously. Result count should be
        # exactly the number of distinct ids ever written, with no
        # `sqlite3.ProgrammingError` or partial-write artefacts.
        await document_store.write_documents_async([_doc("seed")])
        await asyncio.gather(
            document_store.write_documents_async([_doc("a")]),
            document_store.count_documents_async(),
            document_store.write_documents_async([_doc("b")]),
            document_store.filter_documents_async(),
            document_store.write_documents_async([_doc("c")]),
        )
        ids = sorted(d.id for d in await document_store.filter_documents_async())
        assert ids == ["a", "b", "c", "seed"]


class TestQueryByEmbeddingAsync:
    async def test_top_k_ordering(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async(
            [
                _doc("near", embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc("mid", embedding=[0.5, 0.5, 0.0, 0.0]),
                _doc("far", embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        result = await document_store._query_by_embedding_async([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert [d.id for d in result] == ["near", "mid", "far"]

    async def test_with_filter(self, document_store: SQLiteVecDocumentStore):
        await document_store.write_documents_async(
            [
                _doc("a", meta={"lang": "en"}, embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc("b", meta={"lang": "ko"}, embedding=[1.0, 0.0, 0.0, 0.0]),
            ]
        )
        result = await document_store._query_by_embedding_async(
            [1.0, 0.0, 0.0, 0.0],
            top_k=5,
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
        )
        assert [d.id for d in result] == ["b"]
