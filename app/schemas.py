from typing import Any

from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    doc_id: str = Field(..., description="Unique document id")
    text: str = Field(..., min_length=10, description="Document text payload")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    total_chunks_indexed: int


class IngestPdfResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    total_chunks_indexed: int
    file_name: str
    page_count: int
    tables_extracted: int
    images_embedded: int
    ocr_page_count: int
    image_ocr_count: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_generation: bool = True
    # Optional BYOK overrides (API key is sent via X-LLM-Api-Key header, not body).
    llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL (e.g. https://api.openai.com/v1).",
    )
    llm_model: str | None = Field(
        default=None,
        description="Model id for chat completions or responses route.",
    )
    llm_route: str | None = Field(
        default=None,
        description="chat_completions or responses",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    cached: bool = False
    route: str
    retrieved_context: list[RetrievedChunk]


class IndexedDocumentInfo(BaseModel):
    doc_id: str
    chunk_count: int
    file_name: str | None = Field(
        default=None,
        description="Original filename when ingested via PDF (from chunk metadata).",
    )
    source: str | None = Field(
        default=None,
        description="e.g. pdf when from PDF ingest; may be absent for plain /ingest.",
    )


class IndexDocumentsResponse(BaseModel):
    total_chunks: int
    documents: list[IndexedDocumentInfo]


class ClearIndexResponse(BaseModel):
    status: str = "ok"
    indexed_vectors: int = 0
