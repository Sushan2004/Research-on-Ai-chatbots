import math

import torch
from torch import nn


class AttentionHead(nn.Module):
    """One causal attention head with its own Q/K/V projections."""

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.head_dim = head_dim

    def forward(self, x):
        # Each projection: [batch, sequence, head_dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compare every query with every key.
        # Result: [batch, sequence, sequence]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Allow attention to the current position and earlier positions.
        length = x.shape[1]
        allowed = torch.tril(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=x.device,
            )
        )

        # Future positions receive zero probability after softmax.
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        # Combine value vectors using attention probabilities.
        return weights @ v


class MultiHeadAttention(nn.Module):
    """Run multiple attention heads and combine their outputs."""

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError(
                "num_heads must be positive and divide embedding_dim."
            )

        head_dim = embedding_dim // num_heads

        # Register the heads so PyTorch tracks their parameters.
        # Every head sees the full input but learns different projections.
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dim, head_dim)
            for _ in range(num_heads)
        ])

        # Learn how to mix information from the concatenated heads.
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, x):
        # Each head produces [batch, sequence, head_dim].
        head_outputs = [head(x) for head in self.heads]

        # Join the heads along the feature dimension.
        # Result: [batch, sequence, embedding_dim]
        combined = torch.cat(head_outputs, dim=-1)

        return self.output_projection(combined)


def main():
    torch.manual_seed(42)

    # Two sequences, each with five positions and eight input features.
    x = torch.randn(2, 5, 8)

    # Each of the two heads produces four features per position.
    model = MultiHeadAttention(
        embedding_dim=8,
        num_heads=2,
    )

    output = model(x)
    assert output.shape == (2, 5, 8)

    # Check each head's output size.
    with torch.no_grad():
        for head in model.heads:
            assert head(x).shape == (2, 5, 4)

        # Change only the last position.
        changed_x = x.clone()
        changed_x[:, 4, :] += 10.0
        changed_output = model(changed_x)

        # Earlier positions must not see the changed future input.
        torch.testing.assert_close(
            output[:, :4, :],
            changed_output[:, :4, :],
        )

    # Artificial loss to check gradient flow.
    # No optimizer step is performed in this exercise.
    loss = output.square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient: {name}"
        )

    print("Input shape:", x.shape)
    print("Number of heads:", len(model.heads))
    print("Dimensions per head:", model.heads[0].head_dim)
    print("Output shape:", output.shape)
    print("Multi-head attention checks passed.")


if __name__ == "__main__":
    main()