import argparse
import sys
from pathlib import Path

import torch


# Reuse your Stage 5 decoder.
LESSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LESSON_DIR.parent / "05_decoder"))

from decoder import TinyDecoder


@torch.no_grad()
def generate(model, prompt, max_tokens, temperature, top_p, greedy):
    """Generate a continuation using either greedy or sampled decoding."""
    ids = torch.tensor(
        [list(prompt.encode("utf-8"))],
        dtype=torch.long,
    )
    generated_ids = []

    for _ in range(max_tokens):
        # The learned position table supports only max_length positions.
        context = ids[:, -model.max_length:]
        logits, _ = model(context)

        # Scores for the next byte, from the final input position.
        next_logits = logits[:, -1, :]

        if greedy:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            # Lower temperature concentrates probability on likely tokens.
            # Higher temperature spreads probability more broadly.
            probabilities = torch.softmax(
                next_logits / temperature,
                dim=-1,
            )

            # Sort tokens from most to least probable.
            sorted_probs, sorted_ids = torch.sort(
                probabilities,
                descending=True,
                dim=-1,
            )

            # Keep the smallest leading group whose cumulative probability
            # reaches top_p. Retain the token that crosses the threshold.
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False

            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1,
                keepdim=True,
            )

            # Sample an index within the sorted distribution.
            sampled_index = torch.multinomial(
                sorted_probs,
                num_samples=1,
            )

            # Map back to the original vocabulary ID.
            next_id = sorted_ids.gather(-1, sampled_index)

        ids = torch.cat([ids, next_id], dim=1)
        generated_ids.append(next_id.item())

    # Decode bytes together because Unicode characters may span bytes.
    # Invalid generated UTF-8 is shown with replacement characters.
    return bytes(generated_ids).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Always select the highest-scoring token.",
    )
    args = parser.parse_args()

    if not 0 < args.temperature < float("inf"):
        parser.error("--temperature must be finite and greater than zero.")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than zero and at most one.")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    # Load the saved model configuration and weights on CPU.
    saved = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if saved.get("tokenizer") != "utf8_bytes_256":
        raise ValueError("This interface requires the 256-byte tokenizer.")

    model = TinyDecoder(**saved["config"])
    model.load_state_dict(saved["model_state"])
    model.eval()

    # Seed after model construction for repeatable sampling.
    torch.manual_seed(args.seed)

    print(f"Loaded checkpoint at training step {saved['step']}.")
    print(f"Context limit: {model.max_length} byte tokens.")
    print("Mode:", "greedy" if args.greedy else "temperature/top-p sampling")
    print("Enter a text prompt. Each prompt is independent.")
    print("Type /quit or press Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("you: ")

            if prompt.strip() == "/quit":
                break
            if not prompt.strip():
                continue

            continuation = generate(
                model=model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                greedy=args.greedy,
            )

            print("model:", continuation, "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    main()