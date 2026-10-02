import math
import torch


def main():
    torch.manual_seed(42)

    batch_size = 1
    sequence_length = 3
    embedding_dim = 4
    head_dim = 2

    # Example input: three tokens, each represented by four numbers.
    # Shape: [batch, sequence, embedding]
    x = torch.randn(batch_size, sequence_length, embedding_dim)

    # Learnable matrices that project inputs into queries, keys, and values.
    # Shape: [embedding_dim, head_dim]
    w_q = torch.randn(embedding_dim, head_dim, requires_grad=True)
    w_k = torch.randn(embedding_dim, head_dim, requires_grad=True)
    w_v = torch.randn(embedding_dim, head_dim, requires_grad=True)

    def attention(inputs):
        # Each projection has shape [batch, sequence, head_dim].
        q = inputs @ w_q
        k = inputs @ w_k
        v = inputs @ w_v

        # Compare every query with every key using their dot product.
        # [batch, sequence, head_dim] @ [batch, head_dim, sequence]
        # Result: [batch, sequence, sequence]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Allow each token to attend only to itself and earlier tokens.
        length = inputs.shape[1]
        allowed = torch.tril(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=inputs.device,
            )
        )

        # Future positions become -infinity before softmax.
        masked_scores = scores.masked_fill(~allowed, float("-inf"))

        # Each row becomes a probability distribution over key positions.
        weights = torch.softmax(masked_scores, dim=-1)

        # Combine value vectors according to those probabilities.
        # Result: [batch, sequence, head_dim]
        output = weights @ v

        return output, weights, q, k, v, allowed

    output, weights, q, k, v, allowed = attention(x)

    # Check projection and output shapes.
    assert q.shape == (1, 3, 2)
    assert k.shape == (1, 3, 2)
    assert v.shape == (1, 3, 2)
    assert weights.shape == (1, 3, 3)
    assert output.shape == (1, 3, 2)

    # Every attention row must sum to one.
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(batch_size, sequence_length),
    )

    # Future positions must receive zero attention.
    torch.testing.assert_close(
        weights[0][~allowed],
        torch.zeros_like(weights[0][~allowed]),
    )

    # The first token can attend only to itself.
    torch.testing.assert_close(
        weights[0, 0],
        torch.tensor([1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(output[0, 0], v[0, 0])

    # Changing the final input must not affect earlier outputs.
    with torch.no_grad():
        modified_x = x.clone()
        modified_x[:, -1, :] += 10.0
        modified_output, *_ = attention(modified_x)

        torch.testing.assert_close(
            output[:, :-1, :],
            modified_output[:, :-1, :],
        )

    # Use an artificial loss to verify gradient flow.
    # This checks differentiability; it does not train a language model.
    loss = output.square().mean()
    loss.backward()

    for name, matrix in [("Q", w_q), ("K", w_k), ("V", w_v)]:
        assert matrix.grad is not None, f"Missing {name} gradient"
        assert torch.isfinite(matrix.grad).all(), (
            f"Invalid {name} gradient"
        )

    print("Query shape:", q.shape)
    print("Key shape:", k.shape)
    print("Value shape:", v.shape)
    print("\nAttention weights:\n", weights.detach())
    print("\nOutput:\n", output.detach())
    print("\nOutput shape:", output.shape)
    print("Single-head attention checks passed.")


if __name__ == "__main__":
    main()