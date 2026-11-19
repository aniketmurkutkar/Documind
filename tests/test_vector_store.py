import numpy as np
import pytest

from app.services.vector_store import ChunkRecord, FaissStore

DIM = 8


def unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def records(doc_id: str, n: int) -> list[ChunkRecord]:
    return [ChunkRecord(chunk_id=f"{doc_id}-{i}", doc_id=doc_id, text=f"{doc_id} chunk {i}", metadata={}) for i in range(n)]


def store_with(docs: dict[str, int], seed: int = 0) -> tuple[FaissStore, np.ndarray]:
    store = FaissStore(DIM)
    all_vectors = unit_vectors(sum(docs.values()), seed)
    offset = 0
    for doc_id, n in docs.items():
        store.add(all_vectors[offset : offset + n], records(doc_id, n))
        offset += n
    return store, all_vectors


def test_new_store_is_empty():
    store = FaissStore(DIM)

    assert store.total == 0
    assert store.search(unit_vectors(1)[0], top_k=5) == []


def test_add_increases_total_and_keeps_records_in_order():
    store, _ = store_with({"a": 3, "b": 2})

    assert store.total == 5
    assert [r.chunk_id for r in store.records] == ["a-0", "a-1", "a-2", "b-0", "b-1"]


def test_add_with_no_records_is_a_noop():
    store = FaissStore(DIM)
    store.add(np.zeros((0, DIM), dtype=np.float32), [])

    assert store.total == 0


def test_add_rejects_length_mismatch():
    store = FaissStore(DIM)

    with pytest.raises(ValueError, match="length mismatch"):
        store.add(unit_vectors(3), records("a", 2))
    assert store.total == 0
    assert store.records == []


def test_add_rejects_dimension_mismatch():
    store = FaissStore(DIM)
    wrong_dim = np.ones((2, DIM + 1), dtype=np.float32)

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add(wrong_dim, records("a", 2))
    assert store.total == 0
    assert store.records == []


def test_search_returns_results_ordered_by_inner_product():
    store = FaissStore(DIM)
    basis = np.eye(DIM, dtype=np.float32)[:3]
    store.add(basis, records("a", 3))

    query = np.array([0.2, 0.9, 0.5] + [0.0] * (DIM - 3), dtype=np.float32)
    results = store.search(query, top_k=3)

    assert [r.chunk_id for r, _ in results] == ["a-1", "a-2", "a-0"]
    scores = [s for _, s in results]
    assert scores == pytest.approx([0.9, 0.5, 0.2])


def test_search_finds_exact_match_first():
    store, vectors = store_with({"a": 10})

    results = store.search(vectors[7], top_k=3)

    assert results[0][0].chunk_id == "a-7"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(results) == 3


def test_search_top_k_larger_than_index_returns_only_existing_records():
    store, vectors = store_with({"a": 2})

    results = store.search(vectors[0], top_k=10)

    assert len(results) == 2


def test_remove_by_doc_ids_drops_chunks_and_rebuilds_index():
    store, vectors = store_with({"a": 3, "b": 2, "c": 1})

    removed = store.remove_by_doc_ids({"b"})

    assert removed == 2
    assert store.total == 4
    assert {r.doc_id for r in store.records} == {"a", "c"}
    # Remaining vectors must still line up with their records after the rebuild.
    results = store.search(vectors[5], top_k=1)
    assert results[0][0].chunk_id == "c-0"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_remove_by_doc_ids_unknown_or_empty_returns_zero():
    store, _ = store_with({"a": 2})

    assert store.remove_by_doc_ids({"missing"}) == 0
    assert store.remove_by_doc_ids(set()) == 0
    assert store.total == 2


def test_remove_all_docs_leaves_searchable_empty_store():
    store, vectors = store_with({"a": 2, "b": 1})

    assert store.remove_by_doc_ids({"a", "b"}) == 3
    assert store.total == 0
    assert store.search(vectors[0], top_k=3) == []


def test_clear_empties_store_and_allows_reuse():
    store, _ = store_with({"a": 3})

    store.clear()

    assert store.total == 0
    assert store.records == []
    assert store.search(unit_vectors(1)[0], top_k=3) == []

    new_vectors = unit_vectors(2, seed=1)
    store.add(new_vectors, records("b", 2))
    assert store.total == 2
    assert store.search(new_vectors[1], top_k=1)[0][0].chunk_id == "b-1"
