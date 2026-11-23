# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Runs Haystack's shared DocumentStore contract tests against this store."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
)

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


class TestDocumentStoreContract(CountDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest, WriteDocumentsTest):
    @pytest.fixture
    def document_store(self, tmp_path: Path) -> Iterator[SQLiteVecDocumentStore]:
        store = SQLiteVecDocumentStore(db_path=tmp_path / "contract.db", embedding_dim=768, return_embedding=True)
        yield store
        store.close()

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        # Embeddings are stored as float32, so they come back slightly rounded.
        assert len(received) == len(expected)
        by_id = sorted(received, key=lambda d: d.id), sorted(expected, key=lambda d: d.id)
        for got, want in zip(*by_id, strict=True):
            assert replace(got, embedding=None, score=None) == replace(want, embedding=None, score=None)
            if want.embedding is None:
                assert got.embedding is None
            else:
                assert got.embedding == pytest.approx(want.embedding, abs=1e-5)

    def test_write_documents(self, document_store: SQLiteVecDocumentStore) -> None:
        # The default policy is store-specific: here NONE behaves like FAIL.
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc])
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])
