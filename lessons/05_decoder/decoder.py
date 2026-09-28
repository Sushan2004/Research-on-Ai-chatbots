import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# Import the attention implementation you already built.
attention_folder = (
    Path(__file__).resolve().parents[1] / "04_attention"
)
sys.path.insert(0, str(attention_folder))

from multi_head import MultiHeadAttention


class DecoderBlock(nn.Module):
    """Causal attention followed by a feedforward network."""

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim, num_heads
        )

        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        # Normalize before attention, then add the original input.
        # This addition is a residual connection.
        x = x + self.attention(self.attention_norm(x))

        # Apply the same feedforward network independently
        # at every token position, with another residual connection.
        x = x + self.feedforward(self.feedforward_norm(x))

        return x


class TinyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.max_length = max_length

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )
        self.position_embedding = nn.Embedding(
            max_length, embedding_dim
        )

        self.blocks = nn.Sequential(*[
            DecoderBlock(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        # Produce one score for every possible next token.
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        # token_ids shape: [batch, sequence]
        if token_ids.ndim != 2:
            raise ValueError("Expected [batch, sequence] token IDs.")

        _, length = token_ids.shape

        if not 1 <= length <= self.max_length:
            raise ValueError("Sequence length is outside the supported range.")

        positions = torch.arange(
            length, device=token_ids.device
        )

        # Position vectors broadcast across the batch.
        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        x = self.blocks(x)
        x = self.final_norm(x)

        # Shape: [batch, sequence, vocabulary]
        logits = self.output(x)

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("Targets must match the input shape.")

            # Cross-entropy takes raw scores, not softmax probabilities.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        """Generate greedily for this small memorization experiment."""
        for _ in range(max_new_tokens):
            # Keep the input within the learned position table.
            context = token_ids[:, -self.max_length:]

            logits, _ = self(context)

            # Use the final position to predict one new token.
            next_token = logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

            token_ids = torch.cat(
                [token_ids, next_token], dim=1
            )

        return token_ids


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Use character IDs to keep this architecture check self-contained.
    # Our BPE tokenizer can be connected in the later training pipeline.
    text = "hello world\n" * 4
    characters = sorted(set(text))
    stoi = {char: index for index, char in enumerate(characters)}
    itos = {index: char for char, index in stoi.items()}

    data = torch.tensor(
        [stoi[char] for char in text],
        dtype=torch.long,
        device=device,
    )

    # Each target is the token immediately following its input position.
    #
    # Input:  h e l l o ...
    # Target: e l l o   ...
    inputs = data[:-1].unsqueeze(0)
    targets = data[1:].unsqueeze(0)

    model = TinyDecoder(
        vocab_size=len(characters),
        max_length=inputs.shape[1],
    ).to(device)

    # Check output shape and initial loss.
    model.eval()
    with torch.no_grad():
        logits, initial_loss = model(inputs, targets)

        assert logits.shape == (
            1, inputs.shape[1], len(characters)
        )
        assert torch.isfinite(initial_loss)

        # Change the final token. Earlier predictions must stay unchanged.
        changed_inputs = inputs.clone()
        changed_inputs[:, -1] = (
            changed_inputs[:, -1] + 1
        ) % len(characters)

        changed_logits, _ = model(changed_inputs)

        torch.testing.assert_close(
            logits[:, :-1, :],
            changed_logits[:, :-1, :],
            atol=1e-6,
            rtol=1e-5,
        )

    print("Decoder shape and causal checks passed.")
    print(f"Initial loss: {initial_loss.item():.6f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0,
    )

    # Deliberately reuse one example to test overfitting.
    model.train()

    for step in range(301):
        optimizer.zero_grad(set_to_none=True)

        _, loss = model(inputs, targets)
        assert torch.isfinite(loss), "Training loss became invalid."

        loss.backward()

        # Check gradient flow through the assembled model once.
        if step == 0:
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, (
                    f"Missing gradient: {name}"
                )
                assert torch.isfinite(parameter.grad).all(), (
                    f"Invalid gradient: {name}"
                )
            print("Decoder gradient checks passed.")

        optimizer.step()

        if step % 50 == 0:
            print(f"step={step:3d} loss={loss.item():.6f}")

    # Evaluate after the final parameter update.
    model.eval()
    with torch.no_grad():
        logits, final_loss = model(inputs, targets)
        predictions = logits.argmax(dim=-1)
        accuracy = (
            predictions == targets
        ).float().mean().item()

    print(f"\nFinal loss: {final_loss.item():.6f}")
    print(f"Training-token accuracy: {accuracy:.2%}")

    assert final_loss.item() < 0.05, "Not yet overfitted."
    assert accuracy > 0.99, "Some training targets remain incorrect."

    # Start with the first character and generate the rest.
    generated = model.generate(
        data[:1].unsqueeze(0),
        max_new_tokens=len(data) - 1,
    )
    generated_text = "".join(
        itos[token] for token in generated[0].tolist()
    )

    print("\nGenerated text:")
    print(generated_text)

    assert generated_text == text, "Generated sequence differs."
    print("Tiny decoder overfitting checks passed.")


if __name__ == "__main__":
    main()