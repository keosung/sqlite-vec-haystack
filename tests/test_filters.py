# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from haystack import Document
from haystack.errors import FilterError

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore
from haystack_integrations.document_stores.sqlite_vec.filters import (
    _convert_filters_to_sqlite,
    _validate_filters,
)


def _doc(id_: str, *, meta: dict[str, Any] | None = None, embedding: list[float] | None = None) -> Document:
    return Document(
        id=id_,
        content=f"content of {id_}",
        meta=meta or {},
        embedding=embedding or [0.1, 0.2, 0.3, 0.4],
    )


# ---------------------------------------------------------------------------
# Pure conversion tests — no DB involved
# ---------------------------------------------------------------------------


class TestValidate:
    def test_none_ok(self):
        _validate_filters(None)

    def test_empty_dict_ok(self):
        _validate_filters({})

    def test_non_dict_raises(self):
        with pytest.raises(FilterError, match="must be a dictionary"):
            _validate_filters("not a dict")  # type: ignore[arg-type]

    def test_missing_operator_and_field_raises(self):
        with pytest.raises(FilterError, match="top-level dict"):
            _validate_filters({"value": 1})


class TestConvertComparison:
    def test_eq_meta_field(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "==", "value": 1})
        assert sql == "(json_extract(meta_json, ?) = ?)"
        assert params == ["$.x", 1]

    def test_eq_nested_meta(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.a.b.c", "operator": "==", "value": 1})
        assert sql == "(json_extract(meta_json, ?) = ?)"
        assert params == ["$.a.b.c", 1]

    def test_eq_with_none(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "==", "value": None})
        assert sql == "(json_extract(meta_json, ?) IS NULL)"
        assert params == ["$.x"]

    def test_neq_with_none(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "!=", "value": None})
        assert sql == "(json_extract(meta_json, ?) IS NOT NULL)"
        assert params == ["$.x"]

    def test_neq_non_none_uses_is_not(self):
        sql, _ = _convert_filters_to_sqlite({"field": "meta.x", "operator": "!=", "value": 1})
        # `IS NOT` is null-safe in SQLite — `NULL IS NOT 5` evaluates to 1.
        assert sql == "(json_extract(meta_json, ?) IS NOT ?)"

    @pytest.mark.parametrize("op", [">", ">=", "<", "<="])
    def test_range_numeric(self, op: str):
        sql, params = _convert_filters_to_sqlite({"field": "meta.n", "operator": op, "value": 5})
        assert sql == f"(json_extract(meta_json, ?) {op} ?)"
        assert params == ["$.n", 5]

    def test_range_iso_date_string_ok(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.d", "operator": ">=", "value": "2024-01-01T00:00:00"})
        assert sql == "(json_extract(meta_json, ?) >= ?)"
        assert params == ["$.d", "2024-01-01T00:00:00"]

    def test_range_non_iso_string_rejected(self):
        with pytest.raises(FilterError, match="ISO-formatted dates"):
            _convert_filters_to_sqlite({"field": "meta.s", "operator": ">", "value": "abc"})

    def test_range_with_none_short_circuits_false(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.n", "operator": ">", "value": None})
        assert sql == "(0)"
        assert params == []

    def test_range_with_list_rejected(self):
        with pytest.raises(FilterError, match="cannot be a list"):
            _convert_filters_to_sqlite({"field": "meta.n", "operator": ">", "value": [1, 2]})

    def test_in_meta_field(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "in", "value": [1, 2, 3]})
        assert sql == "(json_extract(meta_json, ?) IN (?,?,?))"
        assert params == ["$.x", 1, 2, 3]

    def test_in_empty_list_is_false(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "in", "value": []})
        assert sql == "(0)"
        assert params == []

    def test_not_in_meta_field(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "not in", "value": ["a", "b"]})
        # Null-safe NOT IN: NULL matches.
        assert sql == "(json_extract(meta_json, ?) IS NULL OR json_extract(meta_json, ?) NOT IN (?,?))"
        assert params == ["$.x", "$.x", "a", "b"]

    def test_not_in_empty_list_is_true(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "not in", "value": []})
        assert sql == "(1)"
        assert params == []

    def test_in_with_none_element(self):
        # Python: `None in ['a', None]` is True, so NULL fields must match.
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "in", "value": ["a", None]})
        assert sql == "(json_extract(meta_json, ?) IS NULL OR json_extract(meta_json, ?) IN (?))"
        assert params == ["$.x", "$.x", "a"]

    def test_in_with_only_none(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "in", "value": [None]})
        assert sql == "(json_extract(meta_json, ?) IS NULL)"
        assert params == ["$.x"]

    def test_not_in_with_none_element(self):
        # Python: `None not in ['a', None]` is False, so NULL fields must NOT match,
        # and `'b' NOT IN ('a', NULL)` would be SQL-NULL — the None element has to
        # be split out or every concrete value silently fails to match.
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "not in", "value": ["a", None]})
        assert sql == "(json_extract(meta_json, ?) IS NOT NULL AND json_extract(meta_json, ?) NOT IN (?))"
        assert params == ["$.x", "$.x", "a"]

    def test_not_in_with_only_none(self):
        sql, params = _convert_filters_to_sqlite({"field": "meta.x", "operator": "not in", "value": [None]})
        assert sql == "(json_extract(meta_json, ?) IS NOT NULL)"
        assert params == ["$.x"]

    def test_in_requires_list(self):
        with pytest.raises(FilterError, match="requires a list value"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": "in", "value": 1})

    def test_id_column_field(self):
        sql, params = _convert_filters_to_sqlite({"field": "id", "operator": "==", "value": "doc-1"})
        assert sql == "(doc_id = ?)"
        assert params == ["doc-1"]

    def test_content_column_field(self):
        sql, params = _convert_filters_to_sqlite({"field": "content", "operator": "==", "value": "hello"})
        assert sql == "(content = ?)"
        assert params == ["hello"]

    def test_unsupported_field_raises(self):
        with pytest.raises(FilterError, match="Unsupported field"):
            _convert_filters_to_sqlite({"field": "title", "operator": "==", "value": "x"})

    def test_empty_meta_path_rejected(self):
        with pytest.raises(FilterError, match="Empty meta path"):
            _convert_filters_to_sqlite({"field": "meta.", "operator": "==", "value": 1})

    def test_unknown_comparison_operator_rejected(self):
        with pytest.raises(FilterError, match="Unknown comparison operator"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": "~=", "value": 1})

    @pytest.mark.parametrize("bad_value", [["a", "b"], {"k": "v"}, ("a",)])
    def test_eq_with_unbindable_value_rejected(self, bad_value):
        # sqlite3 can only bind int/float/str/bytes/None; anything else must
        # surface as a FilterError instead of sqlite3.InterfaceError.
        with pytest.raises(FilterError, match="Unsupported filter value"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": "==", "value": bad_value})

    def test_neq_with_unbindable_value_rejected(self):
        with pytest.raises(FilterError, match="Unsupported filter value"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": "!=", "value": {"k": "v"}})

    def test_range_with_dict_value_rejected(self):
        with pytest.raises(FilterError, match="Unsupported filter value"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": ">", "value": {"k": "v"}})

    def test_in_with_unbindable_element_rejected(self):
        with pytest.raises(FilterError, match="Unsupported filter value"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": "in", "value": [1, ["nested"]]})

    def test_missing_value_rejected(self):
        with pytest.raises(FilterError, match="Missing 'value'"):
            _convert_filters_to_sqlite({"field": "meta.x", "operator": "=="})


class TestConvertLogical:
    def test_and_two_conditions(self):
        sql, params = _convert_filters_to_sqlite(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.x", "operator": "==", "value": 1},
                    {"field": "meta.y", "operator": "==", "value": 2},
                ],
            }
        )
        assert sql == "((json_extract(meta_json, ?) = ?) AND (json_extract(meta_json, ?) = ?))"
        assert params == ["$.x", 1, "$.y", 2]

    def test_or_two_conditions(self):
        sql, _ = _convert_filters_to_sqlite(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.x", "operator": "==", "value": 1},
                    {"field": "meta.x", "operator": "==", "value": 2},
                ],
            }
        )
        assert " OR " in sql

    def test_not_single_condition(self):
        sql, params = _convert_filters_to_sqlite(
            {
                "operator": "NOT",
                "conditions": [{"field": "meta.x", "operator": "==", "value": 1}],
            }
        )
        assert sql == "(NOT (json_extract(meta_json, ?) = ?))"
        assert params == ["$.x", 1]

    def test_not_multiple_conditions_treated_as_not_and(self):
        # Haystack's DSL (see FilterDocumentsTest.test_not_operator) defines
        # NOT over a list of conditions as NOT (c1 AND c2 AND ...).
        sql, params = _convert_filters_to_sqlite(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.x", "operator": "==", "value": 1},
                    {"field": "meta.y", "operator": "==", "value": 2},
                ],
            }
        )
        assert sql == "(NOT ((json_extract(meta_json, ?) = ?) AND (json_extract(meta_json, ?) = ?)))"
        assert params == ["$.x", 1, "$.y", 2]

    def test_nested_and_or(self):
        sql, _ = _convert_filters_to_sqlite(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.x", "operator": "==", "value": 1},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.y", "operator": "==", "value": 2},
                            {"field": "meta.y", "operator": "==", "value": 3},
                        ],
                    },
                ],
            }
        )
        assert sql.count("(") == sql.count(")")  # balanced
        assert " AND " in sql
        assert " OR " in sql

    def test_unknown_logical_operator(self):
        with pytest.raises(FilterError, match="Unknown logical operator"):
            _convert_filters_to_sqlite({"operator": "XOR", "conditions": []})

    def test_empty_conditions_list_rejected(self):
        with pytest.raises(FilterError, match="non-empty 'conditions'"):
            _convert_filters_to_sqlite({"operator": "AND", "conditions": []})


# ---------------------------------------------------------------------------
# End-to-end tests against a real SQLite store
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_store(document_store: SQLiteVecDocumentStore) -> SQLiteVecDocumentStore:
    document_store.write_documents(
        [
            _doc("a", meta={"lang": "en", "year": 2024, "tags": ["a", "b"]}),
            _doc("b", meta={"lang": "en", "year": 2023, "tags": ["b"]}),
            _doc("c", meta={"lang": "ko", "year": 2024, "tags": []}),
            _doc("d", meta={"lang": "ko", "year": 2022}),  # no tags field
            _doc("e", meta={"lang": None, "year": 2025}),
        ]
    )
    return document_store


class TestFilterDocumentsE2E:
    def test_eq_string(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "==", "value": "en"})
        assert sorted(d.id for d in result) == ["a", "b"]

    def test_eq_int(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents({"field": "meta.year", "operator": "==", "value": 2024})
        assert sorted(d.id for d in result) == ["a", "c"]

    def test_eq_none(self, populated_store: SQLiteVecDocumentStore):
        # meta.lang is None on 'e', and meta.tags is missing on 'd' (also NULL)
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "==", "value": None})
        assert sorted(d.id for d in result) == ["e"]

    def test_neq_none(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "!=", "value": None})
        assert sorted(d.id for d in result) == ["a", "b", "c", "d"]

    def test_gt_int(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents({"field": "meta.year", "operator": ">", "value": 2023})
        assert sorted(d.id for d in result) == ["a", "c", "e"]

    def test_in_string(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "in", "value": ["en", "ko"]})
        assert sorted(d.id for d in result) == ["a", "b", "c", "d"]

    def test_not_in_string(self, populated_store: SQLiteVecDocumentStore):
        # NULL-safe: 'e' (lang=None) is matched by NOT IN (Haystack convention)
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "not in", "value": ["en"]})
        assert sorted(d.id for d in result) == ["c", "d", "e"]

    def test_in_with_none_element_matches_null_fields(self, populated_store: SQLiteVecDocumentStore):
        # Python semantics: lang in ['en', None] → en docs AND the lang=None doc
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "in", "value": ["en", None]})
        assert sorted(d.id for d in result) == ["a", "b", "e"]

    def test_not_in_with_none_element_excludes_null_fields(self, populated_store: SQLiteVecDocumentStore):
        # Python semantics: lang not in ['en', None] → ko docs only;
        # 'e' (lang=None) is excluded because None IS in the list.
        result = populated_store.filter_documents({"field": "meta.lang", "operator": "not in", "value": ["en", None]})
        assert sorted(d.id for d in result) == ["c", "d"]

    def test_not_with_multiple_conditions(self, populated_store: SQLiteVecDocumentStore):
        # NOT (lang=='en' AND year==2024) — only 'a' is (en, 2024).
        # 'e' (lang=None): SQL NULL AND FALSE = FALSE, NOT FALSE = TRUE → matches,
        # same as Python's not (None == 'en' and 2025 == 2024).
        result = populated_store.filter_documents(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                    {"field": "meta.year", "operator": "==", "value": 2024},
                ],
            }
        )
        assert sorted(d.id for d in result) == ["b", "c", "d", "e"]

    def test_and(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                    {"field": "meta.year", "operator": "==", "value": 2024},
                ],
            }
        )
        assert [d.id for d in result] == ["a"]

    def test_or(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "ko"},
                    {"field": "meta.year", "operator": "==", "value": 2025},
                ],
            }
        )
        assert sorted(d.id for d in result) == ["c", "d", "e"]

    def test_not(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents(
            {
                "operator": "NOT",
                "conditions": [{"field": "meta.lang", "operator": "==", "value": "en"}],
            }
        )
        # 'e' has lang=None, NOT(NULL=...) is NULL → treated as false in WHERE,
        # so it's NOT returned. This matches SQL semantics.
        assert sorted(d.id for d in result) == ["c", "d"]

    def test_filter_by_id_column(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store.filter_documents({"field": "id", "operator": "in", "value": ["a", "c"]})
        assert sorted(d.id for d in result) == ["a", "c"]


class TestQueryByEmbeddingWithFilters:
    def test_filter_narrows_candidate_pool(self, populated_store: SQLiteVecDocumentStore):
        result = populated_store._query_by_embedding(
            [0.1, 0.2, 0.3, 0.4],
            top_k=5,
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
        )
        assert all(d.meta["lang"] == "ko" for d in result)
        assert sorted(d.id for d in result) == ["c", "d"]

    def test_runtime_filter_through_retriever(self, populated_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(document_store=populated_store, top_k=10)
        out = retriever.run(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            filters={"field": "meta.year", "operator": ">=", "value": 2024},
        )
        assert sorted(d.id for d in out["documents"]) == ["a", "c", "e"]

    def test_default_filter_set_in_init(self, populated_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(
            document_store=populated_store,
            top_k=10,
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
        )
        out = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert sorted(d.id for d in out["documents"]) == ["c", "d"]

    def test_runtime_filter_overrides_default(self, populated_store: SQLiteVecDocumentStore):
        retriever = SQLiteVecEmbeddingRetriever(
            document_store=populated_store,
            top_k=10,
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
        )
        out = retriever.run(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )
        assert sorted(d.id for d in out["documents"]) == ["a", "b"]

    def test_selective_filter_returns_correct_top_k(self, document_store: SQLiteVecDocumentStore):
        """Store size >> top_k, but only a few docs match the filter.

        Regression guard: a previous implementation applied vec0's ``k``
        constraint before the SQL filter, which silently returned 0 rows
        when the top-K nearest happened to all be filtered out. The fix
        pushes the filter into a ``rowid IN (...)`` subquery so vec0
        scores only matching candidates.
        """
        # 100 docs, only 5 are tagged ``rare``. The remaining 95 are vector
        # neighbours of the query, so a naive implementation would lose all
        # 5 matching docs to the k-cap on the unfiltered set.
        docs = []
        for i in range(100):
            tag = "rare" if i % 20 == 0 else "common"
            docs.append(
                Document(
                    id=f"d{i}",
                    content=f"c{i}",
                    meta={"tag": tag},
                    embedding=[1.0, 0.0, 0.0, float(i) / 1000.0],
                )
            )
        document_store.write_documents(docs)

        retriever = SQLiteVecEmbeddingRetriever(document_store=document_store, top_k=3)
        out = retriever.run(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.tag", "operator": "==", "value": "rare"},
        )
        ids = [d.id for d in out["documents"]]
        assert len(ids) == 3
        assert all(d.meta["tag"] == "rare" for d in out["documents"])
        # Closest five tagged rare are d0, d20, d40, d60, d80; top 3 by
        # distance starts at d0 (smallest perturbation).
        assert ids == ["d0", "d20", "d40"]
