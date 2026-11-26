# Compare the original 3,000-step and 5,000-step checkpoints without updating weights.
# Check saved losses and greedy samples, then record matched generation comparisons.
# This script intentionally targets those historical runs, not the later 10,000-step model.

"""Compare preserved baseline checkpoints without performing training."""
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from chat import TinyDecoder, generate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lessons/06_training"))
from dataset import load_data, get_batch
from train import evaluate


def main():
    # One thread avoids parallel overhead for this very small CPU model.
    torch.set_num_threads(1)
    prompts = ["ROMEO:", "JULIET:", "The king"]
    runs = ["20261002_123352_952051", "20261010_161237_651555"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "threads": 1,
        "prompts": prompts, "seed": 42,
        "seed_policy": "Reset once per decoding mode; prompts run in listed order.",
        "temperature": 0.8, "top_p": 0.9, "max_new_bytes": 100,
        "evaluation_seed": 123, "evaluation_batches_per_split": 10,
        "dataset_sha256": hashlib.sha256(
            (ROOT / "lessons/06_training/data/input.txt").read_bytes()
        ).hexdigest(),
        "checkpoints": [],
    }
    train_data, val_data = load_data()
    # Require the same architecture so this comparison isolates training progress.
    previous_config = None
    for run in runs:
        path = ROOT / "lessons/06_training/runs" / run / "checkpoint.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["tokenizer"] == "utf8_bytes_256"
        if previous_config is not None:
            assert saved["config"] == previous_config
        previous_config = saved["config"]
        model = TinyDecoder(**saved["config"])
        model.load_state_dict(saved["model_state"], strict=True)
        model.eval()
        # Reject NaN or infinite weights before interpreting any generated output.
        assert all(torch.isfinite(p).all() for p in model.parameters())

        # Match train.py exactly: sample train batches before validation batches.
        torch.manual_seed(123)
        batches = {
            name: [get_batch(data, saved["batch_size"], model.max_length)
                   for _ in range(10)]
            for name, data in [("train", train_data), ("validation", val_data)]
        }
        losses = {name: evaluate(model, values) for name, values in batches.items()}
        # Compare against the last saved log row, allowing tiny floating-point differences.
        with path.with_name("losses.csv").open() as handle:
            logged = list(csv.DictReader(handle))[-1]
        assert int(logged["step"]) == saved["step"]
        for name, loss in losses.items():
            assert abs(loss - float(logged[name + "_loss"])) < 1e-5

        # Compare the same greedy prompt used when the checkpoint was saved.
        continuation = generate(model, "ROMEO:\n", 100, 0.8, 0.9, True)
        assert "ROMEO:\n" + continuation == path.with_name(
            f"sample_{saved['step']:04d}.txt"
        ).read_text(encoding="utf-8")
        # Hashes identify the exact checkpoint bytes used for this result.
        record = {
            "path": str(path.relative_to(ROOT)), "step": saved["step"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config": saved["config"], "batch_size": saved["batch_size"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "losses": losses, "logged_losses_match": True,
            "saved_greedy_sample_matches": True, "outputs": {},
        }
        # Both checkpoints use the same seed and prompt order; randomness advances within a mode.
        for mode in ["greedy", "sampled"]:
            torch.manual_seed(42)
            record["outputs"][mode] = [
                {"prompt": prompt, "continuation": generate(
                    model, prompt, 100, 0.8, 0.9, mode == "greedy"
                )} for prompt in prompts
            ]
        report["checkpoints"].append(record)
        print(json.dumps(record, indent=2), flush=True)

    # Use a fresh directory rather than overwriting an earlier comparison.
    destination = ROOT / "lessons/07_chat/comparisons" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"All reload and baseline checks passed. Results: {output}")


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == "__main__":
    main()
