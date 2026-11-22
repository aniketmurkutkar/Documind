import numpy as np
import pytest

from app.services.retriever import HybridRetriever
from app.services.vector_store import ChunkRecord, FaissStore

DIM = 4
N_FILLERS = 12  # more than the dense candidate pool (max(top_k * 2, 10))
QUERY = "what does ERR_4021 mean"
QUERY_VEC = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _build(alpha: float) -> HybridRetriever:
    """Fillers are close to the query embedding but share no query terms;
    the target only matches by keyword and points the opposite way."""
    store = FaissStore(vector_dim=DIM)
    vectors, records = [], []
    for i in range(N_FILLERS):
        vectors.append(_unit([1.0, 0.05 * (i + 1), 0.0, 0.0]))
        records.append(
            ChunkRecord(f"filler-{i}", "doc", f"Gardening tip {i} about watering tomatoes", {})
        )
    vectors.append(_unit([-1.0, 0.0, 0.0, 0.0]))
    records.append(ChunkRecord("target", "doc", "Error ERR_4021 is raised when disk quota runs out", {}))
    store.add(np.vstack(vectors), records)

    retriever = HybridRetriever(store, alpha=alpha)
    retriever.rebuild_sparse_index()
    return retriever


def _ids(hits):
    return [record.chunk_id for record, _ in hits]


def test_bm25_only_match_is_a_candidate():
    hits = _build(alpha=0.0).hybrid_search(QUERY, QUERY_VEC, top_k=3)

    assert _ids(hits)[0] == "target"
    assert hits[0][1] == pytest.approx(1.0)


def test_bm25_only_match_wins_when_sparse_weight_dominates():
    hits = _build(alpha=0.3).hybrid_search(QUERY, QUERY_VEC, top_k=3)

    assert _ids(hits)[0] == "target"
    assert hits[0][1] == pytest.approx(0.7)


def test_dense_only_weighting_follows_dense_order():
    hits = _build(alpha=1.0).hybrid_search(QUERY, QUERY_VEC, top_k=3)

    # Smallest off-axis component = closest to the query vector.
    assert _ids(hits) == ["filler-0", "filler-1", "filler-2"]
    assert hits[0][1] == pytest.approx(1.0)
    assert "target" not in _ids(hits)


def test_scores_are_normalized_over_shared_candidates():
    hits = dict(
        (record.chunk_id, score)
        for record, score in _build(alpha=0.5).hybrid_search(QUERY, QUERY_VEC, top_k=5)
    )

    # Target: lowest dense (0) + highest BM25 (1). Best filler: highest dense (1)
    # + no BM25 (0). Equal weights give equal scores only if both halves share a scale.
    assert hits["target"] == pytest.approx(0.5)
    assert hits["filler-0"] == pytest.approx(0.5)
    assert all(0.0 <= score <= 1.0 for score in hits.values())


def test_empty_store_returns_nothing():
    retriever = HybridRetriever(FaissStore(vector_dim=DIM))
    retriever.rebuild_sparse_index()

    assert retriever.hybrid_search(QUERY, QUERY_VEC, top_k=3) == []
