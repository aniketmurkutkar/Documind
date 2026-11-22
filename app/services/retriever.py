import re

from rank_bm25 import BM25Okapi

from app.services.vector_store import ChunkRecord, FaissStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


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
        """Weighted dense + BM25 search over the union of both top-N candidate sets.

        Candidates come from the dense top-N *and* the BM25 top-N, so a strong
        keyword match (identifier, error code) can be returned even when its
        embedding isn't close to the query. Both score types are then min-max
        normalized over that same candidate set before being combined with alpha.
        """
        records = self.store.records
        if not records:
            return []

        n_candidates = max(top_k * 2, 10)
        index_of = {r.chunk_id: i for i, r in enumerate(records)}

        candidates: list[int] = []
        seen: set[int] = set()
        for record, _ in self.store.search(query_vector, top_k=n_candidates):
            idx = index_of.get(record.chunk_id)
            if idx is not None and idx not in seen:
                seen.add(idx)
                candidates.append(idx)

        sparse_scores = None
        if self._bm25 is not None:
            sparse_scores = self._bm25.get_scores(_tokenize(query))
            n_sparse = min(len(sparse_scores), len(records))
            sparse_top = sorted(range(n_sparse), key=lambda i: sparse_scores[i], reverse=True)
            for idx in sparse_top[:n_candidates]:
                # Zero means no query term overlap; that's not a keyword match.
                if sparse_scores[idx] <= 0:
                    break
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)

        if not candidates:
            return []

        dense_vals = [float(s) for s in self.store.similarity(query_vector, candidates)]
        sparse_vals = [
            float(sparse_scores[i]) if sparse_scores is not None and i < len(sparse_scores) else 0.0
            for i in candidates
        ]
        dense_norm = _normalize(dense_vals)
        sparse_norm = _normalize(sparse_vals)

        scored = [
            (records[idx], self.alpha * d + (1 - self.alpha) * s)
            for idx, d, s in zip(candidates, dense_norm, sparse_norm)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
