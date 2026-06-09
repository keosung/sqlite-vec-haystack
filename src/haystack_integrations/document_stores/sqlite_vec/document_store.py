# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
SQLiteVecDocumentStore — Haystack 2.x DocumentStore backed by sqlite-vec.

Schema
------
Two tables linked by integer rowid:

    documents               regular SQLite table; doc_id (TEXT) UNIQUE
        id (rowid)          INTEGER PRIMARY KEY AUTOINCREMENT
        doc_id              TEXT UNIQUE NOT NULL          # Haystack Document.id
        content             TEXT NOT NULL
        meta_json           TEXT                          # JSON-serialised meta dict

    vec_items               vec0 virtual table; rowid == documents.id
        embedding           FLOAT[D] distance_metric=cosine

Writes are transactional. Queries return Haystack Document instances reconstructed
from the join `documents JOIN vec_items ON id = rowid`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

import sqlite_vec

logger = logging.getLogger(__name__)

VALID_DISTANCE_METRICS = ("cosine", "l2", "dot")

_VEC_DIM_PATTERN = re.compile(r"FLOAT\[(\d+)\]", re.IGNORECASE)
_VEC_METRIC_PATTERN = re.compile(r"distance_metric\s*=\s*(\w+)", re.IGNORECASE)


def _pack(vec: list[float]) -> bytes:
    """Serialise a float vector to the byte format sqlite-vec expects."""
    return struct.pack(f"{len(vec)}f", *vec)


def _row_to_document(row: tuple) -> Document:
    doc_id, content, meta_json = row[:3]
    meta = json.loads(meta_json) if meta_json else {}
    return Document(id=doc_id, content=content, meta=meta)


class SQLiteVecDocumentStore(DocumentStore):
    """
    SQLite-backed document store with vector search via sqlite-vec.

    :param db_path: path to .db file (created if missing).
    :param embedding_dim: dimensionality of stored embeddings; must match the
        embedder used both at indexing time and query time.
    :param distance_metric: 'cosine' (default), 'l2', or 'dot' — passed to vec0.
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding_dim: int = 768,
        distance_metric: str = "cosine",
    ) -> None:
        if distance_metric not in VALID_DISTANCE_METRICS:
            msg = f"distance_metric must be one of {VALID_DISTANCE_METRICS}, got {distance_metric!r}"
            raise ValueError(msg)
        if embedding_dim <= 0:
            msg = f"embedding_dim must be a positive integer, got {embedding_dim}"
            raise ValueError(msg)

        self.db_path = str(db_path)
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # -- connection management ---------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        """Close the underlying SQLite connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    TEXT UNIQUE NOT NULL,
                content   TEXT NOT NULL,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_documents_doc_id
                ON documents(doc_id);
            """
        )

        existing = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_items'").fetchone()

        if existing is None:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE vec_items USING vec0(
                    embedding FLOAT[{self.embedding_dim}] distance_metric={self.distance_metric}
                )
                """
            )
        else:
            self._validate_existing_vec_table(existing[0] or "")

        conn.commit()

    def _validate_existing_vec_table(self, create_sql: str) -> None:
        """
        Check that an existing vec_items virtual table matches our config.

        sqlite-vec's vec0 bakes the embedding dimension and distance metric
        into the table schema, so silently reusing a mismatched .db file would
        produce hard-to-diagnose errors at query time.
        """
        dim_match = _VEC_DIM_PATTERN.search(create_sql)
        if dim_match is not None:
            existing_dim = int(dim_match.group(1))
            if existing_dim != self.embedding_dim:
                msg = (
                    f"Existing database at {self.db_path!r} was created with "
                    f"embedding_dim={existing_dim}, but this store is configured "
                    f"with embedding_dim={self.embedding_dim}."
                )
                raise ValueError(msg)

        metric_match = _VEC_METRIC_PATTERN.search(create_sql)
        if metric_match is not None:
            existing_metric = metric_match.group(1).lower()
            if existing_metric != self.distance_metric:
                msg = (
                    f"Existing database at {self.db_path!r} uses "
                    f"distance_metric={existing_metric!r}, but this store is "
                    f"configured with distance_metric={self.distance_metric!r}."
                )
                raise ValueError(msg)

    # -- DocumentStore interface --------------------------------------------

    def count_documents(self) -> int:
        cur = self._connect().execute("SELECT COUNT(*) FROM documents")
        return int(cur.fetchone()[0])

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        # v0: no metadata filtering yet. Phase 3 adds Haystack DSL → SQL json_extract.
        if filters:
            msg = "SQLiteVecDocumentStore v0 does not support metadata filters yet"
            raise NotImplementedError(msg)
        rows = self._connect().execute("SELECT doc_id, content, meta_json FROM documents").fetchall()
        return [_row_to_document(r) for r in rows]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        if not documents:
            return 0
        conn = self._connect()
        written = 0
        with conn:
            for doc in documents:
                if doc.embedding is None:
                    msg = f"Document {doc.id!r} has no embedding — embed before writing to SQLiteVecDocumentStore"
                    raise ValueError(msg)
                if len(doc.embedding) != self.embedding_dim:
                    msg = f"Document {doc.id!r} embedding dim {len(doc.embedding)} != store dim {self.embedding_dim}"
                    raise ValueError(msg)

                meta_json = json.dumps(doc.meta) if doc.meta else None

                if policy == DuplicatePolicy.OVERWRITE:
                    row = conn.execute(
                        "INSERT INTO documents(doc_id, content, meta_json) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(doc_id) DO UPDATE SET "
                        "content=excluded.content, meta_json=excluded.meta_json "
                        "RETURNING id",
                        (doc.id, doc.content or "", meta_json),
                    ).fetchone()
                elif policy == DuplicatePolicy.SKIP:
                    row = conn.execute(
                        "INSERT INTO documents(doc_id, content, meta_json) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(doc_id) DO NOTHING "
                        "RETURNING id",
                        (doc.id, doc.content or "", meta_json),
                    ).fetchone()
                    if row is None:
                        continue  # skipped (existing doc)
                else:  # NONE / FAIL — raise on conflict
                    row = conn.execute(
                        "INSERT INTO documents(doc_id, content, meta_json) VALUES (?, ?, ?) RETURNING id",
                        (doc.id, doc.content or "", meta_json),
                    ).fetchone()
                if row is None:
                    continue
                rowid = row[0]
                # vec0 doesn't support ON CONFLICT — DELETE + INSERT is the
                # supported idempotent upsert pattern.
                conn.execute("DELETE FROM vec_items WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    (rowid, _pack(doc.embedding)),
                )
                written += 1
        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        conn = self._connect()
        with conn:
            # The string interpolation here only builds a comma-separated list of `?`
            # placeholders sized by the input — all real values flow through
            # parameter binding, so this is not an injection vector.
            placeholders = ",".join("?" * len(document_ids))
            rows = conn.execute(
                f"SELECT id FROM documents WHERE doc_id IN ({placeholders})",  # noqa: S608
                tuple(document_ids),
            ).fetchall()
            int_ids = [r[0] for r in rows]
            if int_ids:
                qs = ",".join("?" * len(int_ids))
                conn.execute(f"DELETE FROM vec_items WHERE rowid IN ({qs})", int_ids)  # noqa: S608
                conn.execute(f"DELETE FROM documents WHERE id IN ({qs})", int_ids)  # noqa: S608

    # -- KNN query (used by SQLiteVecEmbeddingRetriever) --------------------

    def _query_by_embedding(self, query_embedding: list[float], top_k: int = 10) -> list[Document]:
        if len(query_embedding) != self.embedding_dim:
            msg = f"query embedding dim {len(query_embedding)} != store dim {self.embedding_dim}"
            raise ValueError(msg)
        rows = (
            self._connect()
            .execute(
                """
            SELECT d.doc_id, d.content, d.meta_json, v.distance
            FROM vec_items AS v
            JOIN documents AS d ON d.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
                (_pack(query_embedding), int(top_k)),
            )
            .fetchall()
        )
        out: list[Document] = []
        for doc_id, content, meta_json, distance in rows:
            d = _row_to_document((doc_id, content, meta_json))
            out.append(dataclasses.replace(d, score=float(distance)))
        return out

    # -- Haystack required serialisation -----------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary."""
        return default_to_dict(
            self,
            db_path=self.db_path,
            embedding_dim=self.embedding_dim,
            distance_metric=self.distance_metric,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SQLiteVecDocumentStore:
        """Deserializes the component from a dictionary."""
        return default_from_dict(cls, data)
