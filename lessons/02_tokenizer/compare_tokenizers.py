from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from bpe import train_bpe, encode_bpe, decode_bpe


def main():
    # Use the same training text for both tokenizers.
    training_text = (
        "hello there. hello world. "
        "we are learning how tokenizers work. "
    ) * 10

    # Train our tokenizer: 256 base byte tokens plus 20 learned merges.
    _, custom_merges, custom_vocab = train_bpe(
        training_text,
        num_merges=20,
    )

    # Create a new, untrained Hugging Face BPE tokenizer.
    hf = Tokenizer(models.BPE())

    # Represent input as bytes without adding a leading space.
    # Disable regex splitting to approximate our custom implementation.
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )

    # Reconstruct text from the byte-level token representation.
    hf.decoder = decoders.ByteLevel()

    # Include all 256 byte representations so unseen characters
    # can still be encoded. Target a total vocabulary of 276.
    trainer = trainers.BpeTrainer(
        vocab_size=276,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )

    # Train locally on our text; no pretrained model is downloaded.
    hf.train_from_iterator([training_text], trainer=trainer)

    print("Custom vocabulary:", len(custom_vocab))
    print("Library vocabulary:", hf.get_vocab_size())

    samples = ["hello world", "Hello?", "café 🤖", ""]

    for text in samples:
        # Encode and decode using our implementation.
        custom_ids = encode_bpe(text, custom_merges)
        custom_restored = decode_bpe(custom_ids, custom_vocab)

        # Encoding returns an object containing IDs and token pieces.
        encoded = hf.encode(text)

        # Decode the library's IDs using its own vocabulary.
        restored = hf.decode(encoded.ids)

        # Both tokenizers must reconstruct the original input exactly.
        assert custom_restored == text, (
            f"Custom round-trip failed for {text!r}"
        )
        assert restored == text, (
            f"Library round-trip failed for {text!r}"
        )

        print(f"\nText: {text!r}")
        print(f"Original byte count: {len(text.encode('utf-8'))}")

        print(
            f"Custom: {len(custom_ids)} tokens, "
            f"IDs={custom_ids}"
        )
        print(
            f"Library: {len(encoded.ids)} tokens, "
            f"IDs={encoded.ids}"
        )

        # These pieces use the library's visible byte representation.
        # Some may look unusual, especially spaces and Unicode bytes.
        print(f"Library token pieces: {encoded.tokens}")
        print(f"Decoded: {restored!r}")

    print("\nComparison round-trip checks passed.")


if __name__ == "__main__":
    main()