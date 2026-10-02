"""A minimal byte-level Byte-Pair Encoding (BPE) tokenizer.

Run:
    python lessons/02_tokenizer/bpe.py

This educational implementation learns merges across the entire input text.
It does not implement special tokens or word-based pre-tokenization.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def encode_bytes(text):
    """Convert text into UTF-8 byte IDs, each between 0 and 255."""
    return list(text.encode("utf-8"))


def decode_bytes(token_ids):
    """Decode original byte IDs. This does not accept merged token IDs."""
    return bytes(token_ids).decode("utf-8")


def count_pairs(token_ids):
    """Count all adjacent pairs, including overlapping occurrences."""
    counts = {}

    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge_pair(token_ids, pair, new_id):
    """Replace matching pairs from left to right without overlap."""
    result = []
    i = 0

    while i < len(token_ids):
        if (
            i + 1 < len(token_ids)
            and (token_ids[i], token_ids[i + 1]) == pair
        ):
            result.append(new_id)
            i += 2  # Consume both tokens.
        else:
            result.append(token_ids[i])
            i += 1

    return result


def train_bpe(text, num_merges, verbose=False):
    """Learn merge rules and a token-ID-to-bytes vocabulary."""
    if not isinstance(num_merges, int) or num_merges < 0:
        raise ValueError("num_merges must be a non-negative integer.")

    token_ids = encode_bytes(text)

    # Base tokens cover every possible byte, including bytes not seen
    # in the training text.
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for step in range(num_merges):
        counts = count_pairs(token_ids)

        if not counts:
            break  # No pair exists when fewer than two tokens remain.

        # Choose the most frequent pair.
        # Break frequency ties by selecting the smallest pair, making
        # training deterministic for the same input and settings.
        pair = min(counts, key=lambda p: (-counts[p], p))

        # Byte IDs occupy 0..255. Learned tokens start at 256.
        new_id = 256 + step

        # Remember both the rule and the bytes it represents.
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        token_ids = merge_pair(token_ids, pair, new_id)

        if verbose:
            print(
                f"Merge {step + 1}: {pair} -> {new_id}, "
                f"bytes={vocab[new_id]!r}"
            )

    return token_ids, merges, vocab


def encode_bpe(text, merges):
    """Encode new text using existing rules without retraining."""
    token_ids = encode_bytes(text)

    while len(token_ids) >= 2:
        pairs = count_pairs(token_ids)

        # Only pairs learned during training are eligible.
        available = [pair for pair in pairs if pair in merges]

        if not available:
            break

        # Lower token IDs mean earlier-learned rules.
        # Apply the earliest available rule, NOT the pair currently
        # appearing most frequently in this new text.
        pair = min(available, key=lambda p: merges[p])

        token_ids = merge_pair(token_ids, pair, merges[pair])

    return token_ids


def decode_bpe(token_ids, vocab):
    """Expand tokens into bytes, then decode the complete sequence."""
    chunks = []

    for token_id in token_ids:
        if token_id not in vocab:
            raise ValueError(f"Unknown token ID: {token_id}")
        chunks.append(vocab[token_id])

    # Decode after joining: an individual token may contain only part
    # of a multibyte UTF-8 character.
    return b"".join(chunks).decode("utf-8")


def save_tokenizer(path, merges):
    """Save ordered rules; the byte vocabulary can be rebuilt from them."""
    payload = {
        "format": "minimal-byte-bpe",
        "version": 1,
        "merges": [
            [left, right, new_id]
            for (left, right), new_id in sorted(
                merges.items(), key=lambda item: item[1]
            )
        ],
    }

    Path(path).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_tokenizer(path):
    """Load rules and reconstruct each learned token's bytes."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Invalid tokenizer file.")

    if (
        payload.get("format") != "minimal-byte-bpe"
        or payload.get("version") != 1
    ):
        raise ValueError("Unsupported tokenizer format or version.")

    records = payload.get("merges")
    if not isinstance(records, list):
        raise ValueError("Tokenizer file must contain a merges list.")

    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for expected_id, record in enumerate(records, start=256):
        if (
            not isinstance(record, list)
            or len(record) != 3
            or any(type(value) is not int for value in record)
        ):
            raise ValueError("Invalid merge record.")

        left, right, new_id = record
        pair = (left, right)

        # Each rule may reference only previously defined tokens.
        if (
            new_id != expected_id
            or left not in vocab
            or right not in vocab
            or pair in merges
        ):
            raise ValueError("Invalid merge order or token reference.")

        merges[pair] = new_id
        vocab[new_id] = vocab[left] + vocab[right]

    return merges, vocab


def run_tests():
    """Check each component and complete tokenizer round trips."""
    samples = ["hello", "Hello?", "café", "🤖", "", "你好"]

    for text in samples:
        assert decode_bytes(encode_bytes(text)) == text
    print("Byte encoding checks passed.")

    assert count_pairs([1, 2, 1, 2, 3]) == {
        (1, 2): 2,
        (2, 1): 1,
        (2, 3): 1,
    }
    assert count_pairs([]) == {}
    assert count_pairs([7]) == {}
    assert count_pairs([7, 7, 7]) == {(7, 7): 2}
    print("Pair counting checks passed.")

    assert merge_pair([1, 2, 1, 2, 3], (1, 2), 256) == [256, 256, 3]
    assert merge_pair([7, 7, 7], (7, 7), 256) == [256, 7]
    assert merge_pair([7, 7, 7, 7], (7, 7), 256) == [256, 256]
    assert merge_pair([1, 3], (1, 2), 256) == [1, 3]
    assert merge_pair([7], (7, 7), 256) == [7]
    assert merge_pair([], (1, 2), 256) == []
    print("Pair merging checks passed.")

    ids, merges, vocab = train_bpe("abababab", num_merges=2)
    assert merges == {(97, 98): 256, (256, 256): 257}
    assert ids == [257, 257]
    assert vocab[257] == b"abab"
    assert encode_bpe("abababab", merges) == ids
    assert decode_bpe(ids, vocab) == "abababab"

    # Confirm deterministic tie-breaking: (97, 98) wins the tie.
    _, tied_merges, _ = train_bpe("abc", num_merges=1)
    assert tied_merges == {(97, 98): 256}

    # Training can stop early, or learn no merges at all.
    for text in ["", "a"]:
        result, rules, vocabulary = train_bpe(text, num_merges=10)
        assert rules == {}
        assert decode_bpe(result, vocabulary) == text

    result, rules, _ = train_bpe("abc", num_merges=0)
    assert result == encode_bytes("abc")
    assert rules == {}
    print("BPE training checks passed.")

    # Rules trained on ASCII must still round-trip unseen Unicode.
    for text in samples + ["abab", "ababcab", "new punctuation!?"]:
        encoded = encode_bpe(text, merges)
        assert decode_bpe(encoded, vocab) == text

    # Also test merges learned from multibyte text.
    unicode_text = "café 🤖 café 🤖"
    ids_u, merges_u, vocab_u = train_bpe(unicode_text, 12)
    assert encode_bpe(unicode_text, merges_u) == ids_u
    assert decode_bpe(ids_u, vocab_u) == unicode_text
    print("BPE encoding and decoding checks passed.")

    # Use a temporary file to check persistence without leaving test files.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        save_tokenizer(path, merges_u)
        loaded_merges, loaded_vocab = load_tokenizer(path)

        assert loaded_merges == merges_u
        assert loaded_vocab == vocab_u

        for text in samples:
            encoded = encode_bpe(text, loaded_merges)
            assert decode_bpe(encoded, loaded_vocab) == text

    print("Save/load checks passed.")


def demo():
    """Train a small tokenizer and display its output on new text."""
    training_text = (
        "hello there. hello world. "
        "we are learning how tokenizers work. "
    ) * 10

    _, merges, vocab = train_bpe(training_text, num_merges=20)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Learned merges: {len(merges)}")

    for text in ["hello world", "Hello?", "café 🤖"]:
        ids = encode_bpe(text, merges)
        restored = decode_bpe(ids, vocab)

        print(f"\nText: {text!r}")
        print(f"Byte count: {len(encode_bytes(text))}")
        print(f"BPE token count: {len(ids)}")
        print(f"Token IDs: {ids}")
        print(f"Decoded: {restored!r}")

    # Save next to this script, regardless of the terminal's directory.
    path = Path(__file__).with_name("tokenizer.json")
    save_tokenizer(path, merges)
    print(f"\nTokenizer saved to: {path}")


if __name__ == "__main__":
    run_tests()
    demo()