from app.services.chunking import semantic_like_chunk


def chunk(text: str, chunk_size: int = 50, overlap: int = 10, metadata: dict | None = None):
    return semantic_like_chunk(
        doc_id="doc",
        text=text,
        metadata=metadata if metadata is not None else {"source": "test.txt"},
        chunk_size=chunk_size,
        overlap=overlap,
    )


def test_short_paragraphs_stay_whole():
    paragraphs = ["First paragraph.", "Second one is here.", "Third."]
    chunks = chunk("\n\n".join(paragraphs))

    assert [c.text for c in chunks] == paragraphs
    assert all(c.metadata["chunk_strategy"] == "paragraph" for c in chunks)


def test_paragraph_exactly_chunk_size_stays_whole():
    text = "x" * 50
    chunks = chunk(text, chunk_size=50)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata["chunk_strategy"] == "paragraph"


def test_paragraphs_are_stripped_and_blank_ones_dropped():
    chunks = chunk("  alpha  \n\n   \n\n\nbeta\n")

    assert [c.text for c in chunks] == ["alpha", "beta"]


def test_long_paragraph_uses_sliding_window_with_overlap():
    text = "".join(chr(ord("a") + i % 26) for i in range(120))
    chunk_size, overlap = 50, 10
    chunks = chunk(text, chunk_size=chunk_size, overlap=overlap)

    step = chunk_size - overlap
    assert [c.text for c in chunks] == [text[0:50], text[40:90], text[80:120]]
    assert all(c.metadata["chunk_strategy"] == "sliding_window" for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text[step:] == nxt.text[:overlap]
    # The windows together cover the whole paragraph.
    assert chunks[-1].text.endswith(text[-1])


def test_sliding_window_stops_once_end_is_reached():
    text = "y" * 90  # windows at 0 and 40; the second reaches the end
    chunks = chunk(text, chunk_size=50, overlap=10)

    assert len(chunks) == 2
    assert len(chunks[-1].text) == 50


def test_overlap_not_smaller_than_chunk_size_still_advances():
    text = "z" * 12
    chunks = chunk(text, chunk_size=5, overlap=5)  # step falls back to 1

    assert len(chunks) == 8
    assert all(len(c.text) == 5 for c in chunks)


def test_empty_input_returns_no_chunks():
    assert chunk("") == []
    assert chunk("   \n\n \n\n\t") == []


def test_chunk_ids_are_sequential_and_unique_across_strategies():
    text = "short one\n\n" + "w" * 120 + "\n\nshort two"
    chunks = chunk(text, chunk_size=50, overlap=10)

    ids = [c.chunk_id for c in chunks]
    assert ids == [f"doc-{i}" for i in range(len(chunks))]
    assert len(set(ids)) == len(ids)
    assert all(c.doc_id == "doc" for c in chunks)


def test_metadata_is_copied_not_shared():
    metadata = {"source": "a.pdf"}
    chunks = chunk("one\n\ntwo", metadata=metadata)

    assert metadata == {"source": "a.pdf"}
    assert chunks[0].metadata == {"source": "a.pdf", "chunk_strategy": "paragraph"}
    assert chunks[0].metadata is not chunks[1].metadata
