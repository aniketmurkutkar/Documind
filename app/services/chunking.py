from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def semantic_like_chunk(
    doc_id: str,
    text: str,
    metadata: dict,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """
    Lightweight context-aware chunker.
    Splits by paragraphs first, then applies a sliding window over long paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[TextChunk] = []
    index = 0

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}-{index}",
                    doc_id=doc_id,
                    text=paragraph,
                    metadata={**metadata, "chunk_strategy": "paragraph"},
                )
            )
            index += 1
            continue

        step = max(1, chunk_size - overlap)
        for start in range(0, len(paragraph), step):
            window = paragraph[start : start + chunk_size]
            if not window:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}-{index}",
                    doc_id=doc_id,
                    text=window,
                    metadata={**metadata, "chunk_strategy": "sliding_window"},
                )
            )
            index += 1
            if start + chunk_size >= len(paragraph):
                break

    return chunks
