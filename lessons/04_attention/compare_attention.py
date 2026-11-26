# Validate our attention against PyTorch using identical weights and inputs.
# Matching outputs and gradients supports correctness for these cases, not text quality.

"""Compare our attention outputs and gradients with weight-matched PyTorch."""

import torch
from torch import nn

from multi_head import MultiHeadAttention


def compare(batch_size, length, embedding_dim, num_heads):
    torch.manual_seed(42)
    custom = MultiHeadAttention(embedding_dim, num_heads).double()
    reference = nn.MultiheadAttention(
        embedding_dim, num_heads, dropout=0.0, batch_first=True,
    ).double()

    # PyTorch packs all Q rows, then all K rows, then all V rows.
    # Our implementation stores each head's projections separately.
    with torch.no_grad():
        packed = torch.cat([
            torch.cat([getattr(head, name).weight for head in custom.heads])
            for name in ("query", "key", "value")
        ])
        reference.in_proj_weight.copy_(packed)
        # Custom Q/K/V have no biases, but the output projection does.
        reference.in_proj_bias.zero_()
        reference.out_proj.weight.copy_(custom.output_projection.weight)
        reference.out_proj.bias.copy_(custom.output_projection.bias)

    x = torch.randn(batch_size, length, embedding_dim, dtype=torch.float64,
                    requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    # For nn.MultiheadAttention, True means BLOCK this query-key pair.
    # This is the opposite of our custom variable named "allowed".
    blocked = torch.ones(length, length, dtype=torch.bool).triu(diagonal=1)
    actual = custom(x)
    expected, _ = reference(
        reference_x, reference_x, reference_x,
        attn_mask=blocked, need_weights=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)

    # Use the same upstream gradient to compare backward computations.
    probe = torch.randn_like(actual)
    # The shared probe defines the same scalar objective for both implementations.
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-9, rtol=1e-7)
    packed_grad = torch.cat([
        torch.cat([getattr(head, name).weight.grad for head in custom.heads])
        for name in ("query", "key", "value")
    ])
    torch.testing.assert_close(
        packed_grad, reference.in_proj_weight.grad, atol=1e-9, rtol=1e-7,
    )
    for name in ("weight", "bias"):
        torch.testing.assert_close(
            getattr(custom.output_projection, name).grad,
            getattr(reference.out_proj, name).grad,
            atol=1e-9, rtol=1e-7,
        )

    # Independently check that future inputs cannot change earlier outputs.
    if length > 1:
        with torch.no_grad():
            changed = x.detach().clone()
            split = length // 2
            changed[:, split:] += torch.randn_like(changed[:, split:]) * 10
            torch.testing.assert_close(
                actual[:, :split], custom(changed)[:, :split],
                atol=1e-9, rtol=1e-7,
            )

    error = (actual - expected).detach().abs().max().item()
    print(f"PASS: batch={batch_size}, length={length}, dim={embedding_dim}, "
          f"heads={num_heads}; max output error={error:.3e}")


def main():
    # Include a one-token boundary, one head, and the trained model's sizes.
    for case in [(1, 1, 8, 2), (2, 5, 8, 1), (2, 5, 8, 2), (2, 64, 64, 4)]:
        compare(*case)
    print("All attention output, gradient, and causal checks passed.")


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == "__main__":
    main()
