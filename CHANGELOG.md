# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release.

### Added
- Support for both the Haystack 2.x (≥2.27) and 3.x release lines; CI runs
  the test suite and examples against both.
- `SQLiteVecDocumentStore` implementing the Haystack `DocumentStore` protocol
  (`count_documents`, `filter_documents`, `write_documents`,
  `delete_documents`) on top of `sqlite-vec`'s `vec0` virtual table.
- `SQLiteVecEmbeddingRetriever` Haystack `@component` for KNN retrieval,
  with optional default and per-call metadata filters and a
  `filter_policy` (`replace` / `merge`) matching other Haystack retrievers.
- Metadata filtering through the standard Haystack filter DSL.
  Supported comparison operators: `==`, `!=`, `>`, `>=`, `<`, `<=`,
  `in`, `not in`. Logical operators: `AND`, `OR`, `NOT`.
  Field paths are translated to `json_extract(meta_json, '$.path')` and
  pushed into a `rowid IN (...)` subquery so vec0 only scores
  matching candidates. `None` elements in `in` / `not in` lists and
  `NOT` over multiple conditions follow Python membership / boolean
  semantics, matching the Haystack DSL.
- Async surface: `count_documents_async`, `filter_documents_async`,
  `write_documents_async`, `delete_documents_async`,
  `_query_by_embedding_async`, and `SQLiteVecEmbeddingRetriever.run_async`.
- Thread-safe shared access via an internal `threading.RLock`; benchmarked
  to have no measurable throughput cost vs. a no-op lock.
- `to_dict` / `from_dict` with fully qualified upstream-style type paths
  so pipelines serialize and reload cleanly.
- Schema mismatch detection when reopening an existing `.db` with a
  different `embedding_dim` or `distance_metric`.
- Examples in `examples/`: `quickstart.py`, `async_pipeline.py`,
  `embedding_retrieval.py`.
- Benchmark in `scripts/bench_lock_overhead.py`.
- Test suite (150 tests) covering init validation, schema compatibility,
  CRUD, retrieval, filtering, filter policies, async, concurrency, and
  serialization.
