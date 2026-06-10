# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Quickstart — write, count, retrieve, and filter without external models.

Uses synthetic 4-dimensional embeddings so the example runs anywhere
without downloading a model. The API surface shown here is the same one
you would use with real embeddings produced by any Haystack embedder.

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.sqlite_vec import SQLiteVecEmbeddingRetriever
from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


def main() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "quickstart.db"

        store = SQLiteVecDocumentStore(
            db_path=db_path,
            embedding_dim=4,
            distance_metric="cosine",
        )

        # 1. Write — embeddings are produced upstream (here we use literals)
        docs = [
            Document(
                id="berlin",
                content="Berlin is the capital of Germany.",
                meta={"lang": "en", "year": 2024, "category": "geography"},
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Document(
                id="paris",
                content="Paris is the capital of France.",
                meta={"lang": "en", "year": 2024, "category": "geography"},
                embedding=[0.9, 0.1, 0.0, 0.0],
            ),
            Document(
                id="seoul",
                content="Seoul is the capital of South Korea.",
                meta={"lang": "ko", "year": 2023, "category": "geography"},
                embedding=[0.0, 1.0, 0.0, 0.0],
            ),
        ]
        written = store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        print(f"Wrote {written} documents; store now holds {store.count_documents()}.")

        # 2. Top-k retrieval via the retriever component
        retriever = SQLiteVecEmbeddingRetriever(document_store=store, top_k=2)
        result = retriever.run(query_embedding=[1.0, 0.0, 0.0, 0.0])
        print("\nTop-2 closest to query [1, 0, 0, 0]:")
        for doc in result["documents"]:
            print(f"  {doc.id:8s}  score={doc.score:.4f}  '{doc.content}'")

        # 3. Same query, but constrained to Korean documents only
        result_ko = retriever.run(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.lang", "operator": "==", "value": "ko"},
        )
        print("\nTop-2 closest to query [1, 0, 0, 0], filtered to meta.lang == 'ko':")
        for doc in result_ko["documents"]:
            print(f"  {doc.id:8s}  score={doc.score:.4f}  '{doc.content}'")

        # 4. Filter without retrieval — listing documents that match metadata
        recent_geo = store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.year", "operator": ">=", "value": 2024},
                    {"field": "meta.category", "operator": "==", "value": "geography"},
                ],
            }
        )
        print(f"\n{len(recent_geo)} docs match year>=2024 AND category=='geography':")
        for doc in recent_geo:
            print(f"  {doc.id}: {doc.content}")

        store.close()


if __name__ == "__main__":
    main()
