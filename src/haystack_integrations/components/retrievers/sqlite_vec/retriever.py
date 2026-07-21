# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""SQLiteVecEmbeddingRetriever — Haystack retriever component."""

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


@component
class SQLiteVecEmbeddingRetriever:
    """
    Retrieves top-k documents from a SQLiteVecDocumentStore given a query embedding.

    Pluggable in a Haystack pipeline downstream of any text embedder component
    (e.g. `SentenceTransformersTextEmbedder`, `FastembedTextEmbedder`).
    """

    def __init__(
        self,
        document_store: SQLiteVecDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create a SQLiteVecEmbeddingRetriever.

        :param document_store: an instance of SQLiteVecDocumentStore.
        :param filters: default metadata filters applied to every call. How
            runtime filters passed to `run` interact with these depends on
            ``filter_policy``.
        :param top_k: maximum number of documents to return.
        :param filter_policy: policy to determine how init and runtime filters
            combine. ``REPLACE`` (default): runtime filters replace the init
            filters. ``MERGE``: runtime filters are merged with the init
            filters, with runtime values winning on conflicts.
        :raises ValueError: if ``document_store`` is not a ``SQLiteVecDocumentStore``
            instance, or if ``top_k`` is not a positive integer.
        """
        if not isinstance(document_store, SQLiteVecDocumentStore):
            msg = "document_store must be an instance of SQLiteVecDocumentStore"
            raise ValueError(msg)
        if top_k <= 0:
            msg = f"top_k must be a positive integer, got {top_k}"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SQLiteVecEmbeddingRetriever:
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = SQLiteVecDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        # Pipelines serialized before filter_policy existed won't have the key.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Run the retriever on a query embedding.

        :param query_embedding: dense embedding of the query.
        :param filters: per-call metadata filters. How they combine with the
            filters set at construction time depends on the ``filter_policy``
            chosen at construction time.
        :param top_k: optional per-call override of `top_k` from `__init__`.
        :returns: dictionary with key ``"documents"`` containing the top-k Documents.
        """
        effective_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        k = top_k if top_k is not None else self.top_k
        docs = self.document_store._query_by_embedding(
            query_embedding,
            top_k=k,
            filters=effective_filters,
        )
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously run the retriever on a query embedding.

        Same semantics as `run`, dispatched through the document store's
        async query method so the call does not block the event loop.

        :param query_embedding: dense embedding of the query.
        :param filters: per-call metadata filters. How they combine with the
            filters set at construction time depends on the ``filter_policy``
            chosen at construction time.
        :param top_k: optional per-call override of `top_k` from `__init__`.
        :returns: dictionary with key ``"documents"`` containing the top-k Documents.
        """
        effective_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        k = top_k if top_k is not None else self.top_k
        docs = await self.document_store._query_by_embedding_async(
            query_embedding,
            top_k=k,
            filters=effective_filters,
        )
        return {"documents": docs}
