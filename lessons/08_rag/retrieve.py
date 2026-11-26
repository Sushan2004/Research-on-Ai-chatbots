# Lesson 8a: read local Markdown and split it into overlapping character chunks.
# This prepares evidence for search; it neither embeds passages nor generates answers.
# The current callers read the older documents/ copies, not the root progress report.

from pathlib import Path


def load_documents(folder):
    """Read Markdown files and keep their source filenames."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Documents folder not found: {folder}"
        )

    documents = []

    # Only direct Markdown children are included; this is not a recursive file search.
    for path in sorted(folder.glob("*.md")):
        if not path.is_file():
            continue

        # Accept UTF-8 files with or without a byte-order mark.
        text = path.read_text(encoding="utf-8-sig")

        if not text.strip():
            continue

        documents.append({
            "source": path.name,
            "text": text,
        })

    return documents


def chunk_documents(documents, chunk_size=500, overlap=100):
    """Split text into overlapping chunks measured in characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for document in documents:
        text = document["text"]
        start = 0
        # IDs restart for each document; source + chunk_id identifies a passage.
        chunk_number = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_number,
                "start": start,
                "end": end,
                "text": text[start:end],
            })

            # Avoid creating another chunk once we reach the end.
            if end == len(text):
                break

            # Preserve some context in the following chunk.
            start = end - overlap
            chunk_number += 1

    return chunks


def test_chunking():
    """Check boundaries, overlap, and short inputs."""
    example = [{"source": "test.md", "text": "abcdefghij"}]
    chunks = chunk_documents(example, chunk_size=6, overlap=2)

    assert [chunk["text"] for chunk in chunks] == [
        "abcdef",
        "efghij",
    ]

    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 6
    assert chunks[1]["start"] == 4
    assert chunks[1]["end"] == 10

    # The final two characters of the first chunk are reused.
    assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]

    # Source information must survive chunking.
    assert all(chunk["source"] == "test.md" for chunk in chunks)

    # Short documents produce one chunk.
    short = [{"source": "short.md", "text": "hello"}]
    assert len(chunk_documents(short)) == 1

    # Empty input produces no chunks.
    assert chunk_documents([]) == []
    assert chunk_documents([{"source": "empty.md", "text": ""}]) == []

    # Zero overlap is also supported.
    chunks = chunk_documents(example, chunk_size=6, overlap=0)
    assert [chunk["text"] for chunk in chunks] == [
        "abcdef",
        "ghij",
    ]

    print("Chunking checks passed.")


def main():
    test_chunking()

    folder = Path(__file__).parent / "documents"
    documents = load_documents(folder)

    if not documents:
        raise ValueError(
            f"No nonempty Markdown documents found in {folder}"
        )

    print(f"\nLoaded {len(documents)} documents.")

    for document in documents:
        print(
            f"  {document['source']}: "
            f"{len(document['text'])} characters"
        )

    print("Document loading checks passed.")

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    # Verify chunks match the original document slices.
    source_texts = {
        document["source"]: document["text"]
        for document in documents
    }

    for chunk in chunks:
        original = source_texts[chunk["source"]]
        assert chunk["text"] == original[chunk["start"]:chunk["end"]]
        assert 0 < len(chunk["text"]) <= 500

    print(f"\nCreated {len(chunks)} chunks.")

    # Display a few examples without printing all document content.
    for chunk in chunks[:3]:
        print(
            f"\nSource: {chunk['source']} | "
            f"Chunk: {chunk['chunk_id']} | "
            f"Characters: {chunk['start']}–{chunk['end']}"
        )
        print(repr(chunk["text"][:120]))

    print("\nAll document preparation checks passed.")


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == "__main__":
    main()