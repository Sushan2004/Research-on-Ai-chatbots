# Run five retrieval probes and save ranked passages for manual inspection.
# Expected answers are reference labels, not automatic scoring or abstention rules.
# Running this file replaces retrieval_results.json with the new evaluation output.

from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

from retrieve import load_documents, chunk_documents
from semantic_search import MODEL_NAME, build_index, search


# None marks questions whose answers are absent; it does not make search abstain.
CASES = [
    {
        "question": "How many tokens were in our BPE vocabulary?",
        "expected": "276 tokens: 256 base tokens plus 20 merges.",
    },
    {
        "question": "What vocabulary size does our training model use?",
        "expected": "256 byte tokens.",
    },
    {
        "question": "What context length does our training model use?",
        "expected": "64 byte tokens.",
    },
    {
        "question": "What is the weather tomorrow?",
        "expected": None,
    },
    {
        "question": "What is my favorite food?",
        "expected": None,
    },
]


def main():
    folder = Path(__file__).resolve().parent
    documents = load_documents(folder / "documents")
    chunks = chunk_documents(documents, chunk_size=500, overlap=100)

    if not chunks:
        raise ValueError("No searchable documents found.")

    # Reuse the same embedder for passages and questions so vectors are comparable.
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = build_index(model, chunks)

    records = []

    # Retrieve five passages because a supporting answer may rank below the first three.
    for case in CASES:
        results = search(
            model, index, chunks, case["question"], top_k=5
        )

        print(f"\nQUESTION: {case['question']}")
        print(f"EXPECTED: {case['expected'] or 'Unsupported'}")

        for rank, result in enumerate(results, start=1):
            print(
                f"\n[{rank}] {result['source']} "
                f"| chunk {result['chunk_id']} "
                f"| score {result['score']:.4f}"
            )
            print(result["text"])

        # Save raw evidence and scores for human review, rather than claiming automatic accuracy.
        records.append({
            **case,
            "results": results,
        })

    # This fixed filename overwrites the previous evaluation; preserve old runs separately if needed.
    output = folder / "retrieval_results.json"
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output}")


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == "__main__":
    main()