import re

from rank_bm25 import BM25Okapi

from app.services.vector_store import ChunkRecord, FaissStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class HybridRetriever:
    def __init__(self, store: FaissStore, alpha: float = 0.7):
        self.store = store
        self.alpha = alpha
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    def rebuild_sparse_index(self) -> None:
        self._corpus_tokens = [_tokenize(r.text) for r in self.store.records]
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def hybrid_search(
        self,
        query: str,
        query_vector,
        top_k: int,
    ) -> list[tuple[ChunkRecord, float]]:
        dense_hits = self.store.search(query_vector, top_k=max(top_k * 2, 10))
        if not dense_hits:
            return []

        dense_map = {record.chunk_id: score for record, score in dense_hits}
        sparse_map: dict[str, float] = {}

        if self._bm25 is not None:
            q_tokens = _tokenize(query)
            sparse_scores = self._bm25.get_scores(q_tokens)
            for idx, sparse_score in enumerate(sparse_scores):
                if idx >= len(self.store.records):
                    continue
                sparse_map[self.store.records[idx].chunk_id] = float(sparse_score)

        dense_vals = list(dense_map.values())
        sparse_vals = list(sparse_map.values()) if sparse_map else [0.0]
        dense_min, dense_max = min(dense_vals), max(dense_vals)
        sparse_min, sparse_max = min(sparse_vals), max(sparse_vals)

        def normalize(value: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 0.0
            return (value - lo) / (hi - lo)

        scored: list[tuple[ChunkRecord, float]] = []
        seen: set[str] = set()
        for record, dense_score in dense_hits:
            if record.chunk_id in seen:
                continue
            seen.add(record.chunk_id)
            sparse_score = sparse_map.get(record.chunk_id, 0.0)
            hybrid_score = self.alpha * normalize(dense_score, dense_min, dense_max) + (
                1 - self.alpha
            ) * normalize(sparse_score, sparse_min, sparse_max)
            scored.append((record, hybrid_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
