from pathlib import Path

import torch


def load_data():
    """Load the dataset and split it into training and validation text."""
    path = Path(__file__).parent / "data" / "input.txt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download input.txt into lessons/06_training/data first."
        )

    text = path.read_text(encoding="utf-8")

    if not text:
        raise ValueError("The dataset is empty.")

    # Keep the first 90% for training and the final 10% for validation.
    # Split before creating windows so no window crosses the boundary.
    split = int(len(text) * 0.9)
    train_text = text[:split]
    val_text = text[split:]

    # Use a fixed byte vocabulary: token IDs range from 0 to 255.
    # No tokenizer training is needed for this baseline.
    train_data = torch.tensor(
        list(train_text.encode("utf-8")),
        dtype=torch.long,
    )
    val_data = torch.tensor(
        list(val_text.encode("utf-8")),
        dtype=torch.long,
    )

    return train_data, val_data


def get_batch(data, batch_size, block_size):
    """Sample input sequences and targets shifted one token ahead."""
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive.")

    if data.ndim != 1:
        raise ValueError("Expected a one-dimensional token sequence.")

    if len(data) <= block_size:
        raise ValueError(
            "Data needs at least block_size + 1 tokens."
        )

    # Sample valid starting positions. The upper bound is exclusive.
    # We need one extra token beyond each input window for its target.
    starts = torch.randint(
        len(data) - block_size,
        (batch_size,),
    )

    inputs = []
    targets = []

    for start in starts.tolist():
        # Example: [0, 1, 2, 3]
        x = data[start:start + block_size]

        # The next token at every position: [1, 2, 3, 4]
        y = data[start + 1:start + block_size + 1]

        inputs.append(x)
        targets.append(y)

    # Combine individual windows into [batch_size, block_size] tensors.
    return torch.stack(inputs), torch.stack(targets)


def main():
    # Make the sampled batches repeatable for this test.
    torch.manual_seed(42)

    train_data, val_data = load_data()

    print("Vocabulary size: 256")
    print("Training bytes:", len(train_data))
    print("Validation bytes:", len(val_data))

    for name, data in [
        ("train", train_data),
        ("validation", val_data),
    ]:
        x, y = get_batch(
            data,
            batch_size=4,
            block_size=32,
        )

        assert x.shape == (4, 32)
        assert y.shape == (4, 32)
        assert x.dtype == y.dtype == torch.long

        # Confirm all IDs belong to the byte vocabulary.
        assert ((x >= 0) & (x < 256)).all()
        assert ((y >= 0) & (y < 256)).all()

        # Targets align with the following input tokens.
        torch.testing.assert_close(
            x[:, 1:],
            y[:, :-1],
        )

        print(f"\n{name} input shape:", x.shape)
        print(f"{name} target shape:", y.shape)
        print(f"{name} batch checks passed.")

    # With five tokens and a four-token window, only start=0 is valid.
    # This verifies the exact slice boundaries, including the last target.
    x, y = get_batch(
        torch.arange(5),
        batch_size=1,
        block_size=4,
    )

    assert x.tolist() == [[0, 1, 2, 3]]
    assert y.tolist() == [[1, 2, 3, 4]]

    print("\nBoundary checks passed.")
    print("All dataset checks passed.")


if __name__ == "__main__":
    main()