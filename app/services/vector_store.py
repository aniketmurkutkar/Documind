from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


class FaissStore:
    def __init__(self, vector_dim: int):
        import faiss

        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatIP(vector_dim)
        self.records: list[ChunkRecord] = []
        self._vectors = np.zeros((0, vector_dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, records: list[ChunkRecord]) -> None:
        if len(records) == 0:
            return
        v = np.asarray(vectors, dtype=np.float32)
        if v.shape[0] != len(records):
            raise ValueError("vectors and records length mismatch")
        if v.shape[1] != self.vector_dim:
            raise ValueError("vector dimension mismatch")
        self.records.extend(records)
        self._vectors = np.vstack([self._vectors, v]) if self._vectors.shape[0] else v.copy()
        self.index.add(v)

    def remove_by_doc_ids(self, doc_ids: set[str]) -> int:
        """Drop all chunks for these doc_ids and rebuild the dense index."""
        if not doc_ids or not self.records:
            return 0
        n_before = len(self.records)
        keep = [r for r in self.records if r.doc_id not in doc_ids]
        removed = n_before - len(keep)
        if removed == 0:
            return 0
        mask = np.array([r.doc_id not in doc_ids for r in self.records], dtype=bool)
        self.records = keep
        self._vectors = self._vectors[mask]
        self.index = faiss.IndexFlatIP(self.vector_dim)
        if self._vectors.shape[0] > 0:
            self.index.add(self._vectors)
        return removed

    def clear(self) -> None:
        self.records = []
        self._vectors = np.zeros((0, self.vector_dim), dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.vector_dim)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[ChunkRecord, float]]:
        if self.index.ntotal == 0:
            return []

        q = np.expand_dims(query_vector.astype(np.float32), axis=0)
        scores, indices = self.index.search(q, top_k)
        pairs: list[tuple[ChunkRecord, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            pairs.append((self.records[idx], float(score)))
        return pairs

    @property
    def total(self) -> int:
        return self.index.ntotal
