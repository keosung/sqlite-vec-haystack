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
        content             TEXT                          # nullable: Document.content may be None
        meta_json           TEXT                          # JSON-serialised meta dict

    vec_items               vec0 virtual table; rowid == documents.id
        embedding           FLOAT[D] distance_metric=cosine

Writes are transactional. Queries return Haystack Document instances reconstructed
from the join `documents JOIN vec_items ON id = rowid`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any

from haystack import Document, default_from_dict, default_to_dict, logging
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

import sqlite_vec

from .filters import _convert_filters_to_sqlite, _validate_filters

logger = logging.getLogger(__name__)

VALID_DISTANCE_METRICS = ("cosine", "l2", "dot")

_VEC_DIM_PATTERN = re.compile(r"FLOAT\[(\d+)\]", re.IGNORECASE)
_VEC_METRIC_PATTERN = re.compile(r"distance_metric\s*=\s*(\w+)", re.IGNORECASE)


def _pack(vec: list[float]) -> bytes:
    """Serialise a float vector to the byte format sqlite-vec expects."""
    return struct.pack(f"{len(vec)}f", *vec)


def _row_to_document(doc_id: str, content: str | None, meta_json: str | None) -> Document:
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
        # Serializes all access to the shared connection, which is also used
        # from ``asyncio.to_thread`` worker threads by the ``*_async`` methods.
        # RLock so locked methods can call each other.
        self._lock = threading.RLock()
        self._ensure_schema()

    # -- connection management ---------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # ``check_same_thread=False`` lets the ``*_async`` methods use this
            # connection from ``asyncio.to_thread`` worker threads. ``self._lock``
            # serializes all access, so the connection never sees concurrent use.
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        """Close the underlying SQLite connection if open."""
        with self._lock:
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
                content   TEXT,
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
        # vec0 defaults to L2 when the CREATE statement omits distance_metric,
        # so a missing clause means the table is an L2 table.
        existing_metric = metric_match.group(1).lower() if metric_match else "l2"
        if existing_metric != self.distance_metric:
            msg = (
                f"Existing database at {self.db_path!r} uses "
                f"distance_metric={existing_metric!r}, but this store is "
                f"configured with distance_metric={self.distance_metric!r}."
            )
            raise ValueError(msg)

    # -- DocumentStore interface --------------------------------------------

    def count_documents(self) -> int:
        """Returns the number of documents stored."""
        with self._lock:
            cur = self._connect().execute("SELECT COUNT(*) FROM documents")
            return int(cur.fetchone()[0])

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the given Haystack filters.

        Returned Documents do not carry their embeddings; embeddings live in
        the vec0 virtual table and are only used for retrieval.

        :param filters: Haystack filter dictionary, or ``None`` for all documents.
        :returns: matching Documents.
        :raises FilterError: if the filter dictionary is malformed.
        """
        _validate_filters(filters)
        sql = "SELECT doc_id, content, meta_json FROM documents"
        params: list[Any] = []
        if filters:
            predicate, params = _convert_filters_to_sqlite(filters)
            sql = f"{sql} WHERE {predicate}"
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql, parameters=params)
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        return [_row_to_document(*row) for row in rows]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Writes documents and their embeddings to the store in one transaction.

        Every document must carry an embedding of size ``embedding_dim``.
        Under ``DuplicatePolicy.NONE`` this store behaves like ``FAIL``.

        :param documents: documents to write.
        :param policy: how to handle documents whose id already exists.
        :returns: number of documents written; ``SKIP`` does not count skipped ones.
        :raises ValueError: if the input is not a list of Documents, or an
            embedding is missing or has the wrong dimension.
        :raises DuplicateDocumentError: under ``NONE`` / ``FAIL`` when a
            document id already exists.
        """
        if not isinstance(documents, list):
            msg = f"documents must be a list of Document instances, got {type(documents).__name__}"
            raise ValueError(msg)
        if not documents:
            return 0
        if not all(isinstance(d, Document) for d in documents):
            msg = "documents must be a list of Document instances"
            raise ValueError(msg)
        for doc in documents:
            if doc.embedding is None:
                msg = f"Document {doc.id!r} has no embedding — embed before writing to SQLiteVecDocumentStore"
                raise ValueError(msg)
            if len(doc.embedding) != self.embedding_dim:
                msg = f"Document {doc.id!r} embedding dim {len(doc.embedding)} != store dim {self.embedding_dim}"
                raise ValueError(msg)

        with self._lock:
            return self._write_documents_locked(documents, policy)

    def _write_documents_locked(
        self,
        documents: list[Document],
        policy: DuplicatePolicy,
    ) -> int:
        conn = self._connect()
        written = 0
        with conn:
            for doc in documents:
                meta_json = json.dumps(doc.meta) if doc.meta else None

                if policy == DuplicatePolicy.OVERWRITE:
                    row = conn.execute(
                        "INSERT INTO documents(doc_id, content, meta_json) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(doc_id) DO UPDATE SET "
                        "content=excluded.content, meta_json=excluded.meta_json "
                        "RETURNING id",
                        (doc.id, doc.content, meta_json),
                    ).fetchone()
                elif policy == DuplicatePolicy.SKIP:
                    row = conn.execute(
                        "INSERT INTO documents(doc_id, content, meta_json) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(doc_id) DO NOTHING "
                        "RETURNING id",
                        (doc.id, doc.content, meta_json),
                    ).fetchone()
                    if row is None:
                        continue  # skipped (existing doc)
                else:  # NONE / FAIL — raise on conflict
                    try:
                        row = conn.execute(
                            "INSERT INTO documents(doc_id, content, meta_json) VALUES (?, ?, ?) RETURNING id",
                            (doc.id, doc.content, meta_json),
                        ).fetchone()
                    except sqlite3.IntegrityError as e:
                        msg = f"Document with id {doc.id!r} already exists in the store"
                        raise DuplicateDocumentError(msg) from e
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
        """
        Deletes documents (and their vectors) by Haystack document id.

        Ids that are not present in the store are ignored.

        :param document_ids: ids of the documents to delete.
        """
        if not document_ids:
            return
        with self._lock:
            self._delete_documents_locked(document_ids)

    def _delete_documents_locked(self, document_ids: list[str]) -> None:
        conn = self._connect()
        with conn:
            # Only ``?`` placeholders are interpolated; all values are bound.
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

    def _query_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        if len(query_embedding) != self.embedding_dim:
            msg = f"query embedding dim {len(query_embedding)} != store dim {self.embedding_dim}"
            raise ValueError(msg)
        if top_k <= 0:
            msg = f"top_k must be a positive integer, got {top_k}"
            raise ValueError(msg)
        _validate_filters(filters)

        params: list[Any] = [_pack(query_embedding), int(top_k)]
        if filters:
            # vec0 evaluates ``k = ?`` before any outer WHERE clause, so a
            # post-filter would drop matching documents whenever the
            # unfiltered top-k happens to be rejected by the predicate.
            # ``rowid IN (...)`` is the one constraint vec0 honours as a
            # pre-filter, so the predicate goes into a subquery and vec0
            # scores only the matching candidates.
            # ``predicate`` contains only ``?`` placeholders; values are bound.
            predicate, filter_params = _convert_filters_to_sqlite(filters)
            rowid_clause = f"AND v.rowid IN (SELECT id FROM documents WHERE {predicate})"  # noqa: S608
            sql = (
                "SELECT d.doc_id, d.content, d.meta_json, v.distance "
                "FROM vec_items AS v "
                "JOIN documents AS d ON d.id = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? " + rowid_clause + " ORDER BY v.distance"
            )
            params.extend(filter_params)
        else:
            sql = (
                "SELECT d.doc_id, d.content, d.meta_json, v.distance "
                "FROM vec_items AS v "
                "JOIN documents AS d ON d.id = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? "
                "ORDER BY v.distance"
            )

        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql, parameters=params)
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        out: list[Document] = []
        for doc_id, content, meta_json, distance in rows:
            doc = _row_to_document(doc_id, content, meta_json)
            # sqlite-vec reports NULL distance when the metric is undefined
            # for the pair (e.g. cosine with a zero vector).
            score = float(distance) if distance is not None else None
            out.append(dataclasses.replace(doc, score=score))
        return out

    # -- Async API ----------------------------------------------------------
    # SQLite I/O is blocking either way, so instead of adding an aiosqlite
    # dependency each ``*_async`` method runs its sync counterpart on the
    # default thread pool, keeping ``AsyncPipeline`` callers off the event
    # loop's critical path.

    async def count_documents_async(self) -> int:
        return await asyncio.to_thread(self.count_documents)

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        return await asyncio.to_thread(self.filter_documents, filters)

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        return await asyncio.to_thread(self.write_documents, documents, policy)

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        await asyncio.to_thread(self.delete_documents, document_ids)

    async def _query_by_embedding_async(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._query_by_embedding,
            query_embedding,
            top_k,
            filters,
        )

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
