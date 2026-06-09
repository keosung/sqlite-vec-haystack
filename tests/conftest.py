# SPDX-FileCopyrightText: 2026-present Keoseong Park <keosung.park@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


@pytest.fixture
def document_store(tmp_path: Path) -> SQLiteVecDocumentStore:
    """Fresh on-disk store backed by a temp directory.

    Each test gets its own .db file. ``embedding_dim=4`` keeps the fixture
    cheap; tests that need a different dim should construct their own store.
    """
    return SQLiteVecDocumentStore(db_path=tmp_path / "test.db", embedding_dim=4)
