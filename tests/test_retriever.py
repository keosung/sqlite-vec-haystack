# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


class TestInit:
    def test_defaults(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store)
        assert retriever.top_k == 10
        assert retriever.document_store is document_store

    def test_custom_top_k(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=3)
        assert retriever.top_k == 3

    def test_wrong_document_store_type_raises(self):
        with pytest.raises(ValueError, match="must be an instance of SQLiteVecDocumentStore"):
            SQLiteVecEmbeddingRetriever(document_store="not a store")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_top_k", [0, -1, -100])
    def test_invalid_top_k_raises(self, document_store: SQLiteVecDocumentStore, bad_top_k: int):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=bad_top_k)


class TestRun:
    def test_returns_documents(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [
                Document(id="near", content="x", embedding=[1.0, 0.0, 0.0, 0.0]),
                Document(id="far", content="y", embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=2)
        out = retriever.run(query_embedding=[1.0, 0.0, 0.0, 0.0])
        assert "documents" in out
        assert [d.id for d in out["documents"]] == ["near", "far"]

    def test_top_k_override_at_runtime(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [Document(id=f"d{i}", content="x", embedding=[float(i % 2), float(i), 0.0, 0.0]) for i in range(5)]
        )
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=10)
        out = retriever.run(query_embedding=[0.0, 1.0, 0.0, 0.0], top_k=2)
        assert len(out["documents"]) == 2

    def test_empty_store_returns_empty(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=5)
        out = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert out["documents"] == []

    def test_invalid_runtime_top_k_raises(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=0)


class TestFilterPolicy:
    def test_default_is_replace(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store)
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_string_input_converted(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, filter_policy="merge")
        assert retriever.filter_policy == FilterPolicy.MERGE

    def test_invalid_string_raises(self, document_store: SQLiteVecDocumentStore):
        with pytest.raises(ValueError, match="Unknown FilterPolicy"):
            SQLiteVecEmbeddingRetriever(document_store=document_store, filter_policy="bogus")

    def test_merge_combines_init_and_runtime_filters(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [
                Document(id="a", content="x", meta={"lang": "ko", "year": 2024}, embedding=[1.0, 0.0, 0.0, 0.0]),
                Document(id="b", content="y", meta={"lang": "ko", "year": 2022}, embedding=[1.0, 0.0, 0.0, 0.0]),
                Document(id="c", content="z", meta={"lang": "en", "year": 2024}, embedding=[1.0, 0.0, 0.0, 0.0]),
            ]
        )
        retriever = SQLiteVecEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
            filter_policy=FilterPolicy.MERGE,
        )
        out = retriever.run(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.year", "operator": ">=", "value": 2024},
        )
        # MERGE → lang == 'ko' AND year >= 2024 → only 'a'
        assert [d.id for d in out["documents"]] == ["a"]

    def test_replace_uses_runtime_filters_only(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents(
            [
                Document(id="a", content="x", meta={"lang": "ko", "year": 2024}, embedding=[1.0, 0.0, 0.0, 0.0]),
                Document(id="c", content="z", meta={"lang": "en", "year": 2024}, embedding=[1.0, 0.0, 0.0, 0.0]),
            ]
        )
        retriever = SQLiteVecEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
            filter_policy=FilterPolicy.REPLACE,
        )
        out = retriever.run(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.year", "operator": ">=", "value": 2024},
        )
        # REPLACE → runtime filter only → both 2024 docs regardless of lang
        assert sorted(d.id for d in out["documents"]) == ["a", "c"]


class TestSerialization:
    def test_to_dict_uses_upstream_qualified_path(self, document_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=7)
        d = retriever.to_dict()
        assert d["type"] == (
            "haystack_integrations.components.retrievers.sqlite_vec.retriever.SQLiteVecEmbeddingRetriever"
        )
        assert d["init_parameters"]["top_k"] == 7
        assert d["init_parameters"]["filters"] == {}
        assert d["init_parameters"]["filter_policy"] == "replace"
        # Nested store payload is fully serialized
        nested = d["init_parameters"]["document_store"]
        assert nested["type"].endswith("SQLiteVecDocumentStore")

    def test_round_trip(self, tmp_path: Path):
        store = SQLiteVecDocumentStore(db_path=tmp_path / "round.db", embedding_dim=4, distance_metric="l2")
        store.write_documents([Document(id="a", content="x", embedding=[0.1, 0.2, 0.3, 0.4])])
        retriever = SQLiteVecEmbeddingRetriever(document_store=store, top_k=3, filter_policy=FilterPolicy.MERGE)
        d = retriever.to_dict()
        store.close()

        restored = SQLiteVecEmbeddingRetriever.from_dict(d)
        assert restored.top_k == 3
        assert restored.filter_policy == FilterPolicy.MERGE
        assert isinstance(restored.document_store, SQLiteVecDocumentStore)
        assert restored.document_store.embedding_dim == 4
        assert restored.document_store.distance_metric == "l2"
        assert restored.document_store.count_documents() == 1

    def test_from_dict_without_filter_policy_defaults_to_replace(self, tmp_path: Path):
        # Pipelines serialized before filter_policy existed must still load.
        store = SQLiteVecDocumentStore(db_path=tmp_path / "old.db", embedding_dim=4)
        retriever = SQLiteVecEmbeddingRetriever(document_store=store)
        d = retriever.to_dict()
        store.close()
        del d["init_parameters"]["filter_policy"]

        restored = SQLiteVecEmbeddingRetriever.from_dict(d)
        assert restored.filter_policy == FilterPolicy.REPLACE
