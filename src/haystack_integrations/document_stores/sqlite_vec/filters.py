# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Translate Haystack filter dictionaries to SQLite predicates.

The Haystack filter DSL is the same across DocumentStores — see
https://docs.haystack.deepset.ai/docs/metadata-filtering. This module converts
those dicts into a `(sql_predicate, params)` pair that can be inlined into a
SQLite ``WHERE`` clause built against this integration's schema:

    documents(id INTEGER PRIMARY KEY, doc_id TEXT, content TEXT, meta_json TEXT)

Document-level fields (``id``, ``content``) and ``meta.<key>`` paths
(any depth) are supported. The latter use ``json_extract`` with the JSON
path passed as a bound parameter, so user-supplied field names never reach
the SQL string itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from haystack.errors import FilterError

_COMPARISON_OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "in", "not in"}
_LOGICAL_OPERATORS = {"AND", "OR", "NOT"}

# Types that sqlite3 can bind as query parameters. Anything else (list, dict,
# tuple, ...) would surface as a sqlite3.InterfaceError deep inside execute(),
# so we reject it up front with a FilterError instead.
_BINDABLE_TYPES = (int, float, str, bytes)

# These map the Haystack-side field names to physical columns of the
# ``documents`` table. Anything starting with ``meta.`` is treated as a JSON
# path into ``meta_json`` and is parameterized rather than inlined.
_FIELD_COLUMN_MAP = {
    "id": "doc_id",
    "content": "content",
}


def _validate_filters(filters: dict[str, Any] | None) -> None:
    """
    Shallow check that ``filters`` is a well-formed Haystack filter dict.

    The deeper structural and value-type checks happen during conversion in
    `_convert_filters_to_sqlite`.

    :param filters: filter dictionary to validate, or ``None``.
    :raises FilterError: if ``filters`` is not a dict or is missing the
        required top-level shape.
    """
    if not filters:
        return
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary."
        raise FilterError(msg)
    if "operator" not in filters and "field" not in filters:
        msg = (
            "Invalid filter syntax. The top-level dict must contain either 'operator' "
            "(for logical/comparison) or 'field' (for a comparison condition). See "
            "https://docs.haystack.deepset.ai/docs/metadata-filtering."
        )
        raise FilterError(msg)


def _convert_filters_to_sqlite(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """
    Convert a Haystack filter dict into a SQLite predicate.

    :param filters: Haystack filter dictionary. May be a comparison
        (``{"field": ..., "operator": ..., "value": ...}``) or a logical
        condition (``{"operator": "AND"|"OR"|"NOT", "conditions": [...]}``).
    :returns: a tuple ``(predicate_sql, params)``. The predicate is fully
        parenthesised, so callers can safely concatenate it after
        ``WHERE`` / ``AND``.
    :raises FilterError: on unknown operators or malformed conditions.
    """
    if "field" in filters:
        return _parse_comparison(filters)
    return _parse_logical(filters)


def _parse_logical(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    operator = condition.get("operator")
    if operator not in _LOGICAL_OPERATORS:
        msg = f"Unknown logical operator {operator!r}. Valid operators are: {sorted(_LOGICAL_OPERATORS)}."
        raise FilterError(msg)

    conditions = condition.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        msg = f"Logical operator {operator!r} requires a non-empty 'conditions' list."
        raise FilterError(msg)

    if operator == "NOT":
        if len(conditions) != 1:
            msg = "Logical operator 'NOT' requires exactly one inner condition."
            raise FilterError(msg)
        sub_sql, sub_params = _convert_filters_to_sqlite(conditions[0])
        return f"(NOT {sub_sql})", sub_params

    parts: list[str] = []
    params: list[Any] = []
    for inner in conditions:
        sub_sql, sub_params = _convert_filters_to_sqlite(inner)
        parts.append(sub_sql)
        params.extend(sub_params)
    joined = f" {operator} ".join(parts)
    return f"({joined})", params


def _parse_comparison(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    field = condition.get("field")
    if not isinstance(field, str) or not field:
        msg = f"Missing or invalid 'field' in comparison condition: {condition}."
        raise FilterError(msg)

    operator = condition.get("operator")
    if operator not in _COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator {operator!r}. Valid operators are: {sorted(_COMPARISON_OPERATORS)}."
        raise FilterError(msg)

    if "value" not in condition:
        msg = f"Missing 'value' in comparison condition: {condition}."
        raise FilterError(msg)
    value = condition["value"]

    field_sql, field_params = _resolve_field(field)
    return _build_comparison(field_sql, field_params, operator, value)


def _resolve_field(field: str) -> tuple[str, list[Any]]:
    """
    Resolve a Haystack field name to a SQL expression and any params.

    - ``meta.<key[.subkey...]>`` → ``json_extract(meta_json, ?)`` with
      ``'$.key.subkey'`` bound as a parameter.
    - ``id``, ``content`` → direct column references (no params).

    :param field: the field name as it appears in the filter dict.
    :returns: ``(sql_expr, params)``.
    :raises FilterError: if the field is not one of the supported forms.
    """
    if field.startswith("meta."):
        meta_path = field[len("meta.") :]
        if not meta_path:
            msg = f"Empty meta path in field {field!r}."
            raise FilterError(msg)
        json_path = f"$.{meta_path}"
        return "json_extract(meta_json, ?)", [json_path]

    column = _FIELD_COLUMN_MAP.get(field)
    if column is None:
        msg = f"Unsupported field {field!r}. Use 'meta.<key>' (any depth), 'id', or 'content'."
        raise FilterError(msg)
    return column, []


def _ensure_bindable(value: Any, operator: str) -> None:
    if value is not None and not isinstance(value, _BINDABLE_TYPES):
        msg = (
            f"Unsupported filter value of type {type(value).__name__!r} for operator "
            f"'{operator}'. Supported types: int, float, bool, str, bytes, None."
        )
        raise FilterError(msg)


def _build_comparison(
    field_sql: str,
    field_params: list[Any],
    operator: str,
    value: Any,
) -> tuple[str, list[Any]]:
    if operator == "==":
        if value is None:
            return f"({field_sql} IS NULL)", field_params
        _ensure_bindable(value, operator)
        return f"({field_sql} = ?)", [*field_params, value]

    if operator == "!=":
        if value is None:
            return f"({field_sql} IS NOT NULL)", field_params
        _ensure_bindable(value, operator)
        # ``IS NOT`` in SQLite is null-safe: NULL IS NOT 5 → 1.
        return f"({field_sql} IS NOT ?)", [*field_params, value]

    if operator in {">", ">=", "<", "<="}:
        if value is None:
            # Range comparisons with NULL are undefined; the WHERE clause
            # treats undefined as false, matching the Haystack convention.
            return "(0)", []
        if isinstance(value, list):
            msg = f"Filter value cannot be a list for operator '{operator}'."
            raise FilterError(msg)
        _ensure_bindable(value, operator)
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except (ValueError, TypeError) as exc:
                msg = (
                    f"Cannot compare strings with operator '{operator}'. Strings "
                    "are only comparable if they are ISO-formatted dates."
                )
                raise FilterError(msg) from exc
        return f"({field_sql} {operator} ?)", [*field_params, value]

    if operator == "in":
        if not isinstance(value, list):
            msg = f"Operator 'in' requires a list value, got {type(value).__name__}."
            raise FilterError(msg)
        for item in value:
            _ensure_bindable(item, operator)
        if not value:
            return "(0)", []  # x IN () is always false
        placeholders = ",".join("?" * len(value))
        return f"({field_sql} IN ({placeholders}))", [*field_params, *value]

    # ``not in``
    if not isinstance(value, list):
        msg = f"Operator 'not in' requires a list value, got {type(value).__name__}."
        raise FilterError(msg)
    for item in value:
        _ensure_bindable(item, operator)
    if not value:
        return "(1)", []  # x NOT IN () is always true
    placeholders = ",".join("?" * len(value))
    # SQLite's ``NOT IN`` evaluates to NULL (treated as false) when the LHS is
    # NULL; add ``IS NULL OR`` so documents missing the field still match.
    return (
        f"({field_sql} IS NULL OR {field_sql} NOT IN ({placeholders}))",
        [*field_params, *field_params, *value],
    )
