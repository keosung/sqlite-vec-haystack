# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore
from haystack_integrations.document_stores.sqlite_vec import document_store as ds_module


def _doc(id_: str, *, embedding: list[float] | None = None, meta: dict | None = None) -> Document:
    return Document(id=id_, content=f"content of {id_}", meta=meta or {}, embedding=embedding or [0.1, 0.2, 0.3, 0.4])


class TestInit:
    def test_defaults(self, tmp_path: Path):
        store = SQLiteVecDocumentStore(db_path=tmp_path / "x.db")
        assert store.embedding_dim == 768
        assert store.distance_metric == "cosine"
        assert store.db_path == str(tmp_path / "x.db")

    def test_custom_args(self, tmp_path: Path):
        store = SQLiteVecDocumentStore(db_path=tmp_path / "x.db", embedding_dim=384, distance_metric="l2")
        assert store.embedding_dim == 384
        assert store.distance_metric == "l2"

    @pytest.mark.parametrize("bad_metric", ["cosin", "L2", "", "euclidean"])
    def test_invalid_distance_metric_raises(self, tmp_path: Path, bad_metric: str):
        with pytest.raises(ValueError, match="distance_metric must be one of"):
            SQLiteVecDocumentStore(db_path=tmp_path / "x.db", distance_metric=bad_metric)

    @pytest.mark.parametrize("bad_dim", [0, -1, -128])
    def test_invalid_embedding_dim_raises(self, tmp_path: Path, bad_dim: int):
        with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
            SQLiteVecDocumentStore(db_path=tmp_path / "x.db", embedding_dim=bad_dim)

    def test_path_accepts_str_and_pathlib(self, tmp_path: Path):
        SQLiteVecDocumentStore(db_path=str(tmp_path / "a.db"))
        SQLiteVecDocumentStore(db_path=tmp_path / "b.db")


class TestSchemaCompat:
    def test_reopen_with_same_config_works(self, tmp_path: Path):
        db = tmp_path / "x.db"
        s1 = SQLiteVecDocumentStore(db_path=db, embedding_dim=4)
        s1.write_documents([_doc("d1")])
        s1.close()

        s2 = SQLiteVecDocumentStore(db_path=db, embedding_dim=4)
        assert s2.count_documents() == 1

    def test_reopen_with_different_dim_raises(self, tmp_path: Path):
        db = tmp_path / "x.db"
        SQLiteVecDocumentStore(db_path=db, embedding_dim=4).close()
        with pytest.raises(ValueError, match=r"embedding_dim=4.*embedding_dim=8"):
            SQLiteVecDocumentStore(db_path=db, embedding_dim=8)

    def test_reopen_with_different_metric_raises(self, tmp_path: Path):
        db = tmp_path / "x.db"
        SQLiteVecDocumentStore(db_path=db, embedding_dim=4, distance_metric="cosine").close()
        with pytest.raises(ValueError, match=r"distance_metric='cosine'.*distance_metric='l2'"):
            SQLiteVecDocumentStore(db_path=db, embedding_dim=4, distance_metric="l2")

    def test_external_table_without_metric_is_treated_as_l2(self, tmp_path: Path):
        # vec0 defaults to L2 when distance_metric is omitted from the CREATE
        # statement, so an externally created table must be validated as L2.
        db = tmp_path / "ext.db"
        conn = sqlite3.connect(db)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding FLOAT[4])")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match=r"distance_metric='l2'.*distance_metric='cosine'"):
            SQLiteVecDocumentStore(db_path=db, embedding_dim=4, distance_metric="cosine")

        store = SQLiteVecDocumentStore(db_path=db, embedding_dim=4, distance_metric="l2")
        assert store.count_documents() == 0

    def test_reopen_supports_queries(self, tmp_path: Path):
        db = tmp_path / "x.db"
        s1 = SQLiteVecDocumentStore(db_path=db, embedding_dim=4)
        s1.write_documents([_doc("a", embedding=[1.0, 0.0, 0.0, 0.0])])
        s1.close()

        s2 = SQLiteVecDocumentStore(db_path=db, embedding_dim=4)
        result = s2._query_by_embedding([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert [d.id for d in result] == ["a"]
        assert [d.id for d in s2.filter_documents()] == ["a"]

    def test_double_close_is_noop(self, document_store: SQLiteVecDocumentStore):
        document_store.close()
        document_store.close()


class TestCount:
    def test_empty(self, document_store: SQLiteVecDocumentStore):
        assert document_store.count_documents() == 0

    def test_after_writes(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a"), _doc("b"), _doc("c")])
        assert document_store.count_documents() == 3


class TestWriteDocuments:
    def test_returns_number_written(self, document_store: SQLiteVecDocumentStore):
        assert document_store.write_documents([_doc("a"), _doc("b")]) == 2

    def test_empty_list_returns_zero(self, document_store: SQLiteVecDocumentStore):
        assert document_store.write_documents([]) == 0

    def test_overwrite_idempotent(self, document_store: SQLiteVecDocumentStore):
        docs = [_doc("a"), _doc("b")]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert document_store.count_documents() == 2

    def test_skip_existing(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a")], policy=DuplicatePolicy.SKIP)
        # second write with same id and a different content: should be skipped (count unchanged, content unchanged)
        written = document_store.write_documents(
            [Document(id="a", content="updated", embedding=[0.9, 0.9, 0.9, 0.9])], policy=DuplicatePolicy.SKIP
        )
        assert written == 0
        assert document_store.count_documents() == 1
        result = document_store.filter_documents()
        assert result[0].content == "content of a"

    def test_fail_on_duplicate(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a")], policy=DuplicatePolicy.FAIL)
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([_doc("a")], policy=DuplicatePolicy.FAIL)

    def test_missing_embedding_raises(self, document_store: SQLiteVecDocumentStore):
        with pytest.raises(ValueError, match="has no embedding"):
            document_store.write_documents([Document(id="a", content="x")])

    def test_wrong_dim_raises(self, document_store: SQLiteVecDocumentStore):
        with pytest.raises(ValueError, match="!= store dim"):
            document_store.write_documents([Document(id="a", content="x", embedding=[0.1, 0.2])])

    def test_non_list_input_raises(self, document_store: SQLiteVecDocumentStore):
        with pytest.raises(ValueError, match="must be a list"):
            document_store.write_documents("not a list")  # type: ignore[arg-type]

    def test_list_of_non_documents_raises(self, document_store: SQLiteVecDocumentStore):
        with pytest.raises(ValueError, match="must be a list of Document"):
            document_store.write_documents(["not a doc"])  # type: ignore[list-item]

    def test_none_policy_raises_on_duplicate(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a")], policy=DuplicatePolicy.NONE)
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([_doc("a")], policy=DuplicatePolicy.NONE)

    def test_content_none_round_trip(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([Document(id="a", content=None, embedding=[0.1, 0.2, 0.3, 0.4])])
        result = document_store.filter_documents()
        assert result[0].content is None

    def test_meta_round_trip(self, document_store: SQLiteVecDocumentStore):
        meta = {"source": "wiki", "year": 2024, "tags": ["a", "b"]}
        document_store.write_documents([_doc("a", meta=meta)])
        result = document_store.filter_documents()
        assert result[0].meta == meta

    @pytest.mark.parametrize(
        "meta",
        [
            {"k": "한국어 테스트"},
            {"k": {"nested": {"deep": 1}}},
            {"k": True},
            {"k": ""},
            {"k": 9007199254740992},
            {"k": -1.5},
            {"k": 0},
        ],
    )
    def test_meta_round_trip_tricky_values(self, document_store: SQLiteVecDocumentStore, meta: dict):
        document_store.write_documents([_doc("a", meta=meta)])
        assert document_store.filter_documents()[0].meta == meta


class TestDeleteDocuments:
    def test_delete_subset(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a"), _doc("b"), _doc("c")])
        document_store.delete_documents(["a", "c"])
        ids = sorted(d.id for d in document_store.filter_documents())
        assert ids == ["b"]

    def test_delete_non_existing_is_noop(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a")])
        document_store.delete_documents(["does-not-exist"])
        assert document_store.count_documents() == 1

    def test_delete_empty_list_is_noop(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a")])
        document_store.delete_documents([])
        assert document_store.count_documents() == 1

    def test_delete_chunks_id_lists_beyond_param_limit(
        self, document_store: SQLiteVecDocumentStore, monkeypatch: pytest.MonkeyPatch
    ):
        """Id lists longer than the per-statement bind limit are chunked."""
        monkeypatch.setattr(ds_module, "_MAX_PARAMS_PER_STATEMENT", 2)
        document_store.write_documents([_doc(f"d{i}") for i in range(5)])
        document_store.delete_documents([f"d{i}" for i in range(5)])
        assert document_store.count_documents() == 0


class TestFilterDocuments:
    def test_no_filters_returns_all(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a"), _doc("b")])
        result = document_store.filter_documents()
        assert {d.id for d in result} == {"a", "b"}

    def test_empty_dict_returns_all(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a")])
        # an empty dict is treated as "no filter"
        result = document_store.filter_documents(filters={})
        assert len(result) == 1


class TestQueryByEmbedding:
    def test_top_k_ordering(self, document_store: SQLiteVecDocumentStore):
        # Three docs with increasing distance from query [1,0,0,0]
        document_store.write_documents(
            [
                _doc("near", embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc("mid", embedding=[0.5, 0.5, 0.0, 0.0]),
                _doc("far", embedding=[0.0, 1.0, 0.0, 0.0]),
            ]
        )
        result = document_store._query_by_embedding([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert [d.id for d in result] == ["near", "mid", "far"]
        assert all(d.score is not None for d in result)
        # cosine distance is non-decreasing as we go down the list
        scores = [d.score for d in result]
        assert scores == sorted(scores)

    def test_top_k_cap(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc(f"d{i}", embedding=[float(i % 2), float(i), 0.0, 0.0]) for i in range(5)])
        result = document_store._query_by_embedding([0.0, 1.0, 0.0, 0.0], top_k=2)
        assert len(result) == 2

    def test_wrong_query_dim_raises(self, document_store: SQLiteVecDocumentStore):
        with pytest.raises(ValueError, match="query embedding dim"):
            document_store._query_by_embedding([0.1, 0.2], top_k=1)

    @pytest.mark.parametrize("bad_top_k", [0, -1])
    def test_invalid_top_k_raises(self, document_store: SQLiteVecDocumentStore, bad_top_k: int):
        # Guards the runtime path too: the retriever validates top_k at init
        # but `run(top_k=...)` overrides land here unchecked.
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            document_store._query_by_embedding([0.1, 0.2, 0.3, 0.4], top_k=bad_top_k)

    def test_zero_vector_returns_none_score(self, document_store: SQLiteVecDocumentStore):
        # Cosine distance is undefined against a zero vector; sqlite-vec reports
        # NULL distance and the store surfaces it as score=None.
        document_store.write_documents([_doc("z", embedding=[0.0, 0.0, 0.0, 0.0])])
        result = document_store._query_by_embedding([0.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(result) == 1
        assert result[0].score is None


class TestSerialization:
    def test_to_dict_uses_upstream_qualified_path(self, tmp_path: Path):
        store = SQLiteVecDocumentStore(db_path=tmp_path / "x.db", embedding_dim=384, distance_metric="l2")
        d = store.to_dict()
        assert d["type"] == "haystack_integrations.document_stores.sqlite_vec.document_store.SQLiteVecDocumentStore"
        assert d["init_parameters"]["db_path"] == str(tmp_path / "x.db")
        assert d["init_parameters"]["embedding_dim"] == 384
        assert d["init_parameters"]["distance_metric"] == "l2"

    def test_round_trip_preserves_state(self, document_store: SQLiteVecDocumentStore):
        document_store.write_documents([_doc("a"), _doc("b")])
        d = document_store.to_dict()
        document_store.close()

        restored = SQLiteVecDocumentStore.from_dict(d)
        assert restored.count_documents() == 2
        assert restored.embedding_dim == document_store.embedding_dim
        assert restored.distance_metric == document_store.distance_metric
