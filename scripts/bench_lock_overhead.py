"""Micro-benchmark: does the RLock noticeably slow operations?

We measure three things:

1. Pure lock acquire/release cost (baseline).
2. ``count_documents`` throughput — pure read path, the cheapest DB op
   we have. If the lock adds visible overhead, it shows up loudest here.
3. ``write_documents`` throughput — single thread vs concurrent async.
4. A/B comparison: monkey-patch the store with a no-op lock and re-run
   the same workload to see the real cost.

Run from the package root:
    python3 scripts/bench_lock_overhead.py
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import statistics
import tempfile
import threading
import time
from pathlib import Path

from haystack import Document

from haystack_integrations.document_stores.sqlite_vec import SQLiteVecDocumentStore


def time_block(label: str, fn, *, repeat: int = 5):
    """Run ``fn`` ``repeat`` times, return (median_sec, min_sec) and print."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    median = statistics.median(samples)
    print(f"  {label:40s} median={median * 1000:8.2f} ms  min={min(samples) * 1000:8.2f} ms")
    return median


def lock_baseline(n_ops: int = 1_000_000) -> None:
    lock = threading.RLock()
    t0 = time.perf_counter()
    for _ in range(n_ops):
        with lock:
            pass
    elapsed = time.perf_counter() - t0
    per_op_ns = elapsed / n_ops * 1e9
    print(f"  {'pure RLock acquire+release':40s} {per_op_ns:8.0f} ns/op  ({n_ops:,} ops in {elapsed * 1000:.1f} ms)")


def _docs(n: int, dim: int = 4, prefix: str = "d") -> list[Document]:
    return [Document(id=f"{prefix}-{i}", content=f"c {i}", embedding=[float(i % 7), 0.1, 0.2, 0.3]) for i in range(n)]


def bench_count(store: SQLiteVecDocumentStore, n_ops: int) -> None:
    def loop():
        for _ in range(n_ops):
            store.count_documents()

    time_block(f"count_documents x{n_ops:,}", loop)


def bench_write(store: SQLiteVecDocumentStore, batch_size: int) -> None:
    def loop():
        store.write_documents(_docs(batch_size, prefix=f"w-{time.perf_counter_ns()}"))

    time_block(f"write_documents (single batch x{batch_size})", loop, repeat=3)


def bench_concurrent_async(store: SQLiteVecDocumentStore, batch_size: int, n_batches: int) -> None:
    async def gather_writes():
        await asyncio.gather(*(
            store.write_documents_async(
                _docs(batch_size, prefix=f"a-{i}-{time.perf_counter_ns()}"),
            )
            for i in range(n_batches)
        ))

    label = f"async gather: {n_batches} batches x {batch_size}"

    def loop():
        asyncio.run(gather_writes())

    try:
        time_block(label, loop, repeat=3)
    except sqlite3.Error as e:
        print(f"  {label:40s} CRASHED: {type(e).__name__}: {e}")


@contextlib.contextmanager
def disabled_lock(store: SQLiteVecDocumentStore):
    """Swap in a no-op lock for the duration of the context."""

    class _NoLock:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    original = store._lock
    store._lock = _NoLock()
    try:
        yield
    finally:
        store._lock = original


def main() -> None:
    print("[ Lock baseline ]")
    lock_baseline()

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "bench.db"
        store = SQLiteVecDocumentStore(db_path=db, embedding_dim=4)
        # Seed with some data so count_documents has something to scan.
        store.write_documents(_docs(1000))
        assert store.count_documents() == 1000

        for label, ctx in [
            ("[ With RLock (current behaviour) ]", contextlib.nullcontext()),
            ("[ Lock disabled (control) ]", disabled_lock(store)),
        ]:
            print()
            print(label)
            with ctx:
                bench_count(store, n_ops=20_000)
                bench_write(store, batch_size=500)
                bench_concurrent_async(store, batch_size=125, n_batches=8)


if __name__ == "__main__":
    main()
