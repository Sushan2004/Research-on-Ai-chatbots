import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch

LESSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LESSON_DIR.parent / "05_decoder"))

from decoder import TinyDecoder
from dataset import load_data, get_batch


DEFAULT_CONFIG = {
    "vocab_size": 256,
    "max_length": 64,
    "embedding_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
}

EVAL_INTERVAL = 100
EVAL_BATCHES = 10


@torch.no_grad()
def evaluate(model, batches):
    """Measure loss without changing the weights."""
    was_training = model.training
    model.eval()

    losses = [
        model(x, y)[1].item()
        for x, y in batches
    ]

    model.train(was_training)
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample(model):
    """Keep the prompt and greedy decoding consistent between reports."""
    was_training = model.training
    model.eval()

    prompt = torch.tensor(
        [list("ROMEO:\n".encode("utf-8"))],
        dtype=torch.long,
    )
    generated = model.generate(prompt, max_new_tokens=100)

    text = bytes(generated[0].tolist()).decode(
        "utf-8", errors="replace"
    )

    model.train(was_training)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to an existing checkpoint.pt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total target step count, including completed steps.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    train_data, val_data = load_data()

    # Restore the original model settings when resuming.
    saved = None
    if args.resume is not None:
        saved = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )
        if saved["tokenizer"] != "utf8_bytes_256":
            raise ValueError("This script requires the byte tokenizer.")

    config = saved["config"] if saved else DEFAULT_CONFIG.copy()
    batch_size = saved["batch_size"] if saved else 8
    learning_rate = saved["learning_rate"] if saved else 0.0003
    start_step = saved["step"] if saved else 0

    if args.steps <= start_step:
        raise ValueError(
            f"--steps must exceed the completed step count: {start_step}"
        )

    model = TinyDecoder(**config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    if saved:
        model.load_state_dict(saved["model_state"])

        # Restore AdamW's accumulated optimizer statistics too.
        optimizer.load_state_dict(saved["optimizer_state"])
        print(f"Resuming from step {start_step}.", flush=True)

    # Each continuation gets a new folder, preserving previous results.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = LESSON_DIR / "runs" / run_name
    run_dir.mkdir(parents=True)

    # Recreate the same fixed evaluation batches.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        evaluation_batches = {
            "train": [
                get_batch(train_data, batch_size, config["max_length"])
                for _ in range(EVAL_BATCHES)
            ],
            "validation": [
                get_batch(val_data, batch_size, config["max_length"])
                for _ in range(EVAL_BATCHES)
            ],
        }

    # Restore sampling state AFTER constructing the model.
    # Model initialization consumes random numbers.
    if saved:
        torch.set_rng_state(saved["rng_state"])

    print("Results folder:", run_dir, flush=True)

    with (run_dir / "losses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_loss", "validation_loss"])

        def report(step):
            train_loss = evaluate(model, evaluation_batches["train"])
            val_loss = evaluate(model, evaluation_batches["validation"])

            print(
                f"step={step:4d} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={val_loss:.4f}",
                flush=True,
            )
            writer.writerow([step, train_loss, val_loss])
            file.flush()

            sample = generate_sample(model)
            (run_dir / f"sample_{step:04d}.txt").write_text(
                sample, encoding="utf-8"
            )
            print("Sample:", repr(sample), flush=True)

        # Show the restored model before any additional updates.
        report(start_step)
        model.train()

        for step in range(start_step + 1, args.steps + 1):
            x, y = get_batch(
                train_data, batch_size, config["max_length"]
            )

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()

            if step % EVAL_INTERVAL == 0 or step == args.steps:
                report(step)

                checkpoint = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "tokenizer": "utf8_bytes_256",
                    "rng_state": torch.get_rng_state(),
                    "resumed_from": (
                        str(args.resume.resolve())
                        if args.resume else None
                    ),
                }

                # Write a temporary file before replacing the checkpoint.
                temporary_path = run_dir / "checkpoint.tmp"
                torch.save(checkpoint, temporary_path)
                temporary_path.replace(run_dir / "checkpoint.pt")

    # Verify the saved weights reproduce the final output.
    reloaded = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    restored_model = TinyDecoder(**reloaded["config"])
    restored_model.load_state_dict(reloaded["model_state"])

    model.eval()
    restored_model.eval()
    test_x, _ = evaluation_batches["validation"][0]

    with torch.no_grad():
        original_logits, _ = model(test_x)
        restored_logits, _ = restored_model(test_x)

    torch.testing.assert_close(original_logits, restored_logits)

    print("\nCheckpoint reload check passed.")
    print("Completed optimizer steps:", reloaded["step"])
    print("Results saved to:", run_dir)


if __name__ == "__main__":
    main()