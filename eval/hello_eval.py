"""Smoke test: run DocuMind's hybrid retrieval through Inspect on a toy corpus.

Checks that the pipeline, the Inspect solver and the scorer are wired together
before the real dataset (#12) exists. No LLM key is needed: retrieval only, and
`mockllm` satisfies Inspect's required --model flag.

Run from the repo root (WSL / Linux, inside the venv):

    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DIM=384 PYTHONPATH=$PWD \
        inspect eval eval/hello_eval.py --model mockllm/model

Then browse the results with `inspect view`.
"""

from functools import lru_cache

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, TaskState, solver

from app.config import get_settings
from app.schemas import IngestDocument
from app.services.pipeline import RagPipeline

# Small corpus with distractors: one question needs exact keyword matching
# (BM25), the others need paraphrase matching (dense embeddings).
CORPUS = [
    IngestDocument(
        doc_id="errors",
        text=(
            "Error ERR_4021 is raised when the disk quota runs out.\n\n"
            "Error ERR_5003 means the upstream service timed out."
        ),
    ),
    IngestDocument(
        doc_id="warranty",
        text=(
            "The X200 pump is covered for 24 months from the date of purchase.\n\n"
            "Seals and filters are consumables and are not covered."
        ),
    ),
    IngestDocument(
        doc_id="maintenance",
        text=(
            "Replace the X200 intake filter every 500 operating hours.\n\n"
            "Flush the pump with clean water before storing it for winter."
        ),
    ),
]

SAMPLES = [
    Sample(id="keyword-error-code", input="What does ERR_4021 mean?", target="disk quota"),
    Sample(id="paraphrase-warranty", input="How long is the X200 under guarantee?", target="24 months"),
    Sample(id="paraphrase-filter", input="How often should the intake filter be changed?", target="500 operating hours"),
]


@lru_cache(maxsize=1)
def _pipeline() -> RagPipeline:
    pipeline = RagPipeline(get_settings())
    pipeline.ingest(CORPUS)
    return pipeline


@solver
def documind_retrieve(top_k: int = 3):
    """Retrieve the top-k chunks for the question; no answer generation."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _, _, _, retrieved = await _pipeline().query(
            state.input_text, top_k=top_k, use_generation=False
        )
        state.metadata["retrieved"] = [
            {"chunk_id": c.chunk_id, "score": round(c.score, 4), "text": c.text}
            for c in retrieved
        ]
        state.output = ModelOutput.from_content(
            model="documind-retriever",
            content="\n\n---\n\n".join(c.text for c in retrieved),
        )
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def evidence_retrieved():
    """CORRECT if the target evidence text appears in any retrieved chunk."""

    async def score(state: TaskState, target: Target) -> Score:
        needle = target.text.lower()
        chunks = state.metadata.get("retrieved", [])
        rank = next(
            (i + 1 for i, c in enumerate(chunks) if needle in c["text"].lower()), None
        )
        return Score(
            value=CORRECT if rank else INCORRECT,
            answer=chunks[0]["text"] if chunks else "",
            explanation=f"evidence at rank {rank}" if rank else "evidence not retrieved",
            metadata={"rank": rank},
        )

    return score


@task
def documind_hello(top_k: int = 3):
    # Load the model and build the index up front, so download and indexing
    # time isn't charged to whichever sample happens to run first.
    _pipeline()
    return Task(
        dataset=SAMPLES,
        solver=documind_retrieve(top_k=top_k),
        scorer=evidence_retrieved(),
    )
