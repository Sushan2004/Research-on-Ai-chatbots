# Lesson 8b: retrieve document passages with pretrained MiniLM embeddings and FAISS.
# This model embeds text; it is separate from our custom next-byte decoder.
# The index is rebuilt in memory on startup. Results are passages, not generated answers.

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieve import load_documents, chunk_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(model, texts):
    """Convert texts into normalized vectors suitable for FAISS."""
    vectors = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 1,
    )

    # FAISS expects contiguous float32 arrays.
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numbers.")

    return vectors


def build_index(model, chunks):
    """Embed each chunk and add its vector to an exact search index."""
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(model, texts)

    # Check that normalization produced unit-length vectors.
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-5,
    )

    # IP means inner product. For unit vectors this is cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    assert index.ntotal == len(chunks)

    # Searching for an indexed vector should find a score near 1.
    # We check the score because duplicate passages may tie.
    scores, _ = index.search(vectors[:1], 1)
    assert np.isclose(scores[0, 0], 1.0, atol=1e-5)

    return index


def search(model, index, chunks, question, top_k=3):
    """Return the closest passages with their scores and sources."""
    if not question.strip():
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Use the SAME model and normalization as the document embeddings.
    question_vector = embed_texts(model, [question])

    # Nearest neighbors are returned even for unsupported questions; no rejection threshold exists.
    scores, positions = index.search(
        question_vector,
        min(top_k, len(chunks)),
    )

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        chunk = chunks[int(position)]

        results.append({
            **chunk,
            "score": float(score),
        })

    return results


def main():
    folder = Path(__file__).parent / "documents"
    documents = load_documents(folder)
    chunks = chunk_documents(documents, chunk_size=500, overlap=100)

    if not chunks:
        raise ValueError(f"No searchable text found in {folder}")

    print(f"Loaded {len(documents)} documents and {len(chunks)} chunks.")
    print("Loading embedding model...", flush=True)

    # The first run downloads model files. Embedding runs locally on CPU.
    # Pretrained semantic embeddings come from MiniLM, not our Shakespeare decoder.
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    index = build_index(model, chunks)

    print(f"\nIndexed {index.ntotal} chunks.")
    print(f"Embedding dimensions: {index.d}")
    print("Index checks passed.")
    print("\nAsk a question to retrieve passages. Type /quit to exit.")
    print("Scores measure similarity, not whether an answer is correct.")

    while True:
        try:
            question = input("\nquestion: ")

            if question.strip() == "/quit":
                break
            if not question.strip():
                continue

            results = search(model, index, chunks, question, top_k=5)

            for rank, result in enumerate(results, start=1):
                print(
                    f"\n[{rank}] {result['source']} "
                    f"| chunk {result['chunk_id']} "
                    f"| similarity {result['score']:.4f}"
                )
                print(
                    f"Character range: "
                    f"{result['start']}–{result['end']}"
                )
                print(result["text"])

        except (KeyboardInterrupt, EOFError):
            print("\nSearch ended.")
            break


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == "__main__":
    main()