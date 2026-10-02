import math

import torch
from torch import nn


def sinusoidal_positions(sequence_length, embedding_dim):
    """Create fixed positional vectors using sine and cosine."""
    if embedding_dim % 2 != 0:
        raise ValueError("This example requires an even embedding dimension.")

    # Shape: [sequence_length, 1]
    positions = torch.arange(
        sequence_length, dtype=torch.float32
    ).unsqueeze(1)

    # Each pair of embedding dimensions uses a different frequency.
    frequencies = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embedding_dim)
    )

    # Broadcasting produces shape [sequence_length, embedding_dim / 2].
    angles = positions * frequencies

    encoding = torch.zeros(sequence_length, embedding_dim)

    # Even columns use sine; odd columns use cosine.
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)

    return encoding


def main():
    torch.manual_seed(42)

    vocab_size = 10
    embedding_dim = 4
    max_length = 8

    # Two sequences containing three token IDs each.
    token_ids = torch.tensor(
        [[4, 2, 4],
         [1, 0, 3]],
        dtype=torch.long,
    )
    sequence_length = token_ids.shape[1]

    # --------------------------------------------------
    # 1. Token embeddings
    # --------------------------------------------------

    # A learnable table with 10 rows and 4 columns.
    token_embedding = nn.Embedding(vocab_size, embedding_dim)

    # Lookup adds an embedding dimension: [2, 3] -> [2, 3, 4].
    token_vectors = token_embedding(token_ids)

    assert token_vectors.shape == (2, 3, 4)

    # Repeated token IDs retrieve the same row.
    torch.testing.assert_close(
        token_vectors[0, 0], token_vectors[0, 2]
    )
    torch.testing.assert_close(
        token_vectors[0, 0], token_embedding.weight[4]
    )

    print("Token embedding table:", token_embedding.weight.shape)
    print("Token vectors:", token_vectors.shape)
    print("Token embedding checks passed.")

    # --------------------------------------------------
    # 2. Learned positional embeddings
    # --------------------------------------------------

    # A separate trainable vector for each supported position.
    position_embedding = nn.Embedding(max_length, embedding_dim)

    if sequence_length > max_length:
        raise ValueError("Sequence exceeds the position table size.")

    positions = torch.arange(sequence_length, dtype=torch.long)
    learned_positions = position_embedding(positions)

    # [2, 3, 4] + [3, 4]: positions broadcast across the batch.
    learned_combined = token_vectors + learned_positions

    assert learned_positions.shape == (3, 4)
    assert learned_combined.shape == (2, 3, 4)

    # Token 4 now has different representations at positions 0 and 2.
    assert not torch.allclose(
        learned_combined[0, 0], learned_combined[0, 2]
    )

    print("\nPosition IDs:", positions)
    print("Learned position vectors:\n", learned_positions)
    print("Learned position checks passed.")

    # --------------------------------------------------
    # 3. Sinusoidal positional encoding
    # --------------------------------------------------

    # These vectors are calculated, rather than learned.
    fixed_positions = sinusoidal_positions(
        sequence_length, embedding_dim
    )
    fixed_combined = token_vectors + fixed_positions

    assert fixed_positions.shape == (3, 4)
    assert fixed_combined.shape == (2, 3, 4)
    assert not fixed_positions.requires_grad

    # At position zero: sin(0)=0 and cos(0)=1.
    torch.testing.assert_close(
        fixed_positions[0],
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    assert not torch.allclose(
        fixed_combined[0, 0], fixed_combined[0, 2]
    )

    print("\nSinusoidal position vectors:\n", fixed_positions)
    print("Sinusoidal position checks passed.")

    # --------------------------------------------------
    # 4. Verify gradients reach the learned tables
    # --------------------------------------------------

    token_embedding.zero_grad()
    position_embedding.zero_grad()

    # Artificial scalar objective to inspect gradient flow.
    # This is not a language-model training loss.
    probe_loss = learned_combined.sum()
    probe_loss.backward()

    token_grad = token_embedding.weight.grad
    position_grad = position_embedding.weight.grad

    assert token_grad is not None
    assert position_grad is not None

    # Token 4 occurs twice, so each component receives gradient 2.
    torch.testing.assert_close(
        token_grad[4], torch.full((embedding_dim,), 2.0)
    )

    # Token 9 never appears, so its row receives no gradient.
    torch.testing.assert_close(
        token_grad[9], torch.zeros(embedding_dim)
    )

    # Each position occurs once in each of the two batch sequences.
    torch.testing.assert_close(
        position_grad[:sequence_length],
        torch.full((sequence_length, embedding_dim), 2.0),
    )

    # Unused positions receive no gradient.
    torch.testing.assert_close(
        position_grad[sequence_length:],
        torch.zeros(max_length - sequence_length, embedding_dim),
    )

    print("\nGradient checks passed.")
    print("Token-table parameters:", token_embedding.weight.numel())
    print("Position-table parameters:", position_embedding.weight.numel())
    print("\nAll embedding exercise checks passed.")


if __name__ == "__main__":
    main()