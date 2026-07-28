from collections import defaultdict

from app.config import Settings
from app.schemas import IndexedDocumentInfo, IndexDocumentsResponse, RetrievedChunk
from app.services.chunking import semantic_like_chunk
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.retriever import HybridRetriever
from app.services.vector_store import ChunkRecord, FaissStore


class RagPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = EmbeddingService(settings.embedding_model)
        self.store = FaissStore(settings.vector_dim)
        self.retriever = HybridRetriever(self.store, alpha=settings.hybrid_alpha)
        api_key = (
            settings.llm_api_key
            or settings.openai_api_key
            or settings.groq_api_key
        ).strip()
        self.llm = LLMService(
            api_key=api_key,
            model=settings.openai_model,
            cache_ttl_seconds=settings.prompt_cache_ttl_seconds,
            cache_max_size=settings.prompt_cache_max_size,
            base_url=settings.llm_base_url,
            llm_route=settings.llm_route,
        )

    def ingest(self, documents) -> tuple[int, int, int]:
        doc_ids = {d.doc_id for d in documents}
        self.store.remove_by_doc_ids(doc_ids)
        self.retriever.rebuild_sparse_index()

        all_chunks = []
        for doc in documents:
            chunks = semantic_like_chunk(
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return len(documents), 0, self.store.total

        vectors = self.embeddings.embed_texts([chunk.text for chunk in all_chunks])
        records = [
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in all_chunks
        ]
        self.store.add(vectors=vectors, records=records)
        self.retriever.rebuild_sparse_index()
        return len(documents), len(records), self.store.total

    def clear_index(self) -> None:
        self.store.clear()
        self.retriever.rebuild_sparse_index()

    def list_indexed_documents(self) -> IndexDocumentsResponse:
        by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
        for r in self.store.records:
            by_doc[r.doc_id].append(r)
        docs: list[IndexedDocumentInfo] = []
        for doc_id in sorted(by_doc.keys()):
            recs = by_doc[doc_id]
            md0 = recs[0].metadata or {}
            fn = md0.get("file_name")
            file_name = fn if isinstance(fn, str) else None
            src = md0.get("source")
            source = src if isinstance(src, str) else None
            docs.append(
                IndexedDocumentInfo(
                    doc_id=doc_id,
                    chunk_count=len(recs),
                    file_name=file_name,
                    source=source,
                )
            )
        return IndexDocumentsResponse(total_chunks=self.store.total, documents=docs)

    async def query(
        self,
        query: str,
        top_k: int,
        use_generation: bool,
        *,
        client_api_key: str | None = None,
        client_base_url: str | None = None,
        client_model: str | None = None,
        client_llm_route: str | None = None,
    ):
        q_vec = self.embeddings.embed_query(query)
        hits = self.retriever.hybrid_search(query=query, query_vector=q_vec, top_k=top_k)

        contexts = []
        retrieved = []
        for record, score in hits:
            contexts.append(record.text)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=record.chunk_id,
                    doc_id=record.doc_id,
                    score=score,
                    text=record.text,
                    metadata=record.metadata,
                )
            )

        joined_context = "\n\n---\n\n".join(contexts) if contexts else "No context retrieved."
        if not use_generation:
            return joined_context, False, "retrieve-only", retrieved

        llm = self._llm_for_request(
            client_api_key=client_api_key,
            client_base_url=client_base_url,
            client_model=client_model,
            client_llm_route=client_llm_route,
        )
        answer, cached, route = await llm.answer(query=query, context=joined_context)
        return answer, cached, route, retrieved

    def _llm_for_request(
        self,
        *,
        client_api_key: str | None,
        client_base_url: str | None,
        client_model: str | None,
        client_llm_route: str | None,
    ) -> LLMService:
        key = (client_api_key or "").strip()
        if key:
            base = (client_base_url or self.settings.llm_base_url).strip()
            model = (client_model or self.settings.openai_model).strip()
            route = (client_llm_route or self.settings.llm_route).strip()
            return LLMService(
                api_key=key,
                model=model,
                cache_ttl_seconds=self.settings.prompt_cache_ttl_seconds,
                cache_max_size=self.settings.prompt_cache_max_size,
                base_url=base,
                llm_route=route,
            )
        return self.llm
