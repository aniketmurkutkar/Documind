import asyncio
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import (
    ClearIndexResponse,
    IngestDocument,
    IngestRequest,
    IngestResponse,
    IngestPdfResponse,
    IndexDocumentsResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.document_extract import extract_pdf_bytes
from app.services.pipeline import RagPipeline

settings = get_settings()

_pipeline: RagPipeline | None = None
_pipeline_lock = Lock()
_pipeline_ready = False
_pipeline_error: str | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline, _pipeline_ready, _pipeline_error
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            _pipeline = RagPipeline(settings)
            _pipeline_ready = True
            _pipeline_error = None
            return _pipeline
        except Exception as exc:  # noqa: BLE001 — surface load failures to callers
            _pipeline_error = str(exc)
            raise


def _warm_pipeline() -> None:
    try:
        get_pipeline()
    except Exception:  # noqa: BLE001 — error stored in _pipeline_error
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load BAAI/bge-m3 in the background so /health can pass Railway checks immediately.
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _warm_pipeline)
    yield


app = FastAPI(title="Documind RAG API", version="0.1.0", lifespan=lifespan)

_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    indexed = _pipeline.store.total if _pipeline is not None else 0
    return {
        "status": "ok",
        "model_ready": _pipeline_ready,
        "model_error": _pipeline_error,
        "indexed_vectors": indexed,
    }


@app.get("/index/documents", response_model=IndexDocumentsResponse)
def list_index_documents() -> IndexDocumentsResponse:
    return get_pipeline().list_indexed_documents()


@app.post("/index/clear", response_model=ClearIndexResponse)
def clear_index() -> ClearIndexResponse:
    pipe = get_pipeline()
    pipe.clear_index()
    return ClearIndexResponse(indexed_vectors=pipe.store.total)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    docs, chunks, total = get_pipeline().ingest(payload.documents)
    return IngestResponse(
        documents_ingested=docs,
        chunks_created=chunks,
        total_chunks_indexed=total,
    )


def _is_pdf_name(name: str | None) -> bool:
    if not name:
        return False
    return name.lower().endswith(".pdf")


@app.post("/ingest/pdf", response_model=IngestPdfResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    doc_id: str | None = Form(
        default=None,
        description="Document id; defaults to uploaded filename (stem).",
    ),
) -> IngestPdfResponse:
    if not _is_pdf_name(file.filename):
        if file.content_type not in ("application/pdf", "application/x-pdf"):
            raise HTTPException(
                status_code=400,
                detail="File must be a .pdf (or content-type application/pdf).",
            )
    data = await file.read()
    max_b = int(settings.pdf_max_size_mb * 1024 * 1024)
    if len(data) > max_b:
        raise HTTPException(
            status_code=413,
            detail=f"PDF larger than {settings.pdf_max_size_mb} MB limit.",
        )
    name = file.filename or "document.pdf"
    d_id = doc_id or (name.rsplit(".", 1)[0] if "." in name else name)

    extracted = extract_pdf_bytes(data, name, settings)
    document = IngestDocument(
        doc_id=d_id,
        text=extracted.text,
        metadata=extracted.metadata,
    )
    docs, chunks, total = get_pipeline().ingest([document])
    s = extracted.stats
    return IngestPdfResponse(
        documents_ingested=docs,
        chunks_created=chunks,
        total_chunks_indexed=total,
        file_name=name,
        page_count=s.page_count,
        tables_extracted=s.tables_extracted,
        images_embedded=s.images_found,
        ocr_page_count=s.ocr_pages,
        image_ocr_count=s.image_ocr_runs,
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-Api-Key"),
) -> QueryResponse:
    top_k = payload.top_k or settings.top_k
    answer, cached, route, retrieved = await get_pipeline().query(
        query=payload.query,
        top_k=top_k,
        use_generation=payload.use_generation,
        client_api_key=x_llm_api_key,
        client_base_url=payload.llm_base_url,
        client_model=payload.llm_model,
        client_llm_route=payload.llm_route,
    )
    return QueryResponse(
        answer=answer,
        cached=cached,
        route=route,
        retrieved_context=retrieved,
    )
