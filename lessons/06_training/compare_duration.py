# Experiment: extend the recorded 64-byte context model from 5,000 to 10,000 updates.
# Restore the optimizer and random state as well as weights to continue the same run.
# Paths deliberately select the recorded baseline; this is not a general training CLI.
# Running this program performs training and saves results in a new directory.

"""Continue the context experiment's 64-byte model with its original sampler."""
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from train import TinyDecoder, evaluate, generate_sample
from dataset import load_data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'lessons/07_chat'))
from chat import generate


def main():
    # Match the previous experiment's CPU execution setting.
    torch.set_num_threads(1)
    source = Path(__file__).parent / 'experiments/context_20261114_185438_026390'
    path = source / '64/checkpoint.pt'
    # A file fingerprint lets us later check that the source checkpoint was preserved.
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    prior = json.loads((source / 'results.json').read_text())
    dataset_hash = hashlib.sha256((ROOT / 'lessons/06_training/data/input.txt').read_bytes()).hexdigest()
    # Continuing on changed data would no longer isolate training duration.
    assert dataset_hash == prior['dataset_sha256']
    saved = torch.load(path, weights_only=True, map_location='cpu')
    assert saved['step'] == 5000 and saved['config']['max_length'] == 64
    assert saved['tokenizer'] == 'utf8_bytes_256'
    train_data, val_data = load_data()

    def batch(data, ends):
        # Keep the original endpoint sampler: inputs end at e-1, targets end at e.
        return (torch.stack([data[e-64:e] for e in ends.tolist()]),
                torch.stack([data[e-63:e+1] for e in ends.tolist()]))

    # Reproduce the context experiment's fixed evaluation endpoints exactly.
    torch.manual_seed(123)
    endpoints = {name: torch.randint(128, len(data), (10, 8)) for name, data in [('train', train_data), ('validation', val_data)]}
    # 128 batches of 8 endpoints provide 1,024 matched final-byte evaluation targets.
    common = torch.randint(128, len(val_data), (128, 8))
    batches = {name: [batch(data, ends) for ends in endpoints[name]] for name, data in [('train', train_data), ('validation', val_data)]}
    model = TinyDecoder(**saved['config'])
    model.load_state_dict(saved['model_state'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=saved['learning_rate'])
    # Restore AdamW's accumulated statistics, not just the model's learned weights.
    optimizer.load_state_dict(saved['optimizer_state'])
    # Reproduce the previous measurement before allowing any additional training.
    baseline_losses = {name: evaluate(model, values) for name, values in batches.items()}
    for name, loss in baseline_losses.items():
        assert abs(loss - prior['results'][0]['losses'][name]) < 1e-6
    # Restore RNG only after constructing the model and fixed evaluation batches.
    torch.set_rng_state(saved['rng_state'])
    output = source.parent / datetime.now(timezone.utc).strftime('duration_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    # Snapshot the implementation alongside results so later edits do not erase provenance.
    for src in [Path(__file__), Path(__file__).with_name('train.py'), Path(__file__).with_name('dataset.py'), ROOT / 'lessons/05_decoder/decoder.py', ROOT / 'lessons/04_attention/multi_head.py', ROOT / 'lessons/07_chat/chat.py']:
        (output / src.name).write_bytes(src.read_bytes())
    print('Experiment:', output, flush=True)
    start = time.perf_counter()
    with (output / 'losses.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'train_loss', 'validation_loss'])
        writer.writerow([5000, baseline_losses['train'], baseline_losses['validation']])
        # These are 5,000 additional updates; the final total is 10,000, not 15,000.
        for step in range(5001, 10001):
            model.train()
            x, y = batch(train_data, torch.randint(128, len(train_data), (8,)))
            # Start this update with no accumulated gradients from the previous batch.
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            # Compute gradients, bound their norm, then let AdamW update the parameters.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            # Record held-out loss and free-running text; neither is used as a training target.
            if step % 500 == 0:
                losses = {name: evaluate(model, values) for name, values in batches.items()}
                writer.writerow([step, losses['train'], losses['validation']])
                handle.flush()
                (output / f'sample_{step:05d}.txt').write_text(generate_sample(model))
                # The sampler label documents a difference from the original train.py sampler.
                checkpoint = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), config=saved['config'], step=step, batch_size=8, learning_rate=saved['learning_rate'], tokenizer=saved['tokenizer'], rng_state=torch.get_rng_state(), resumed_from=str(path.resolve()), sampler='endpoint_uniform_128_to_length_exclusive')
                # Preserve the original baseline; only replace a checkpoint inside this new run.
                torch.save(checkpoint, output / 'checkpoint.tmp')
                (output / 'checkpoint.tmp').replace(output / 'checkpoint.pt')
                print(f'step={step} train={losses["train"]:.4f} validation={losses["validation"]:.4f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    duration = time.perf_counter() - start
    # Confirm the saved final model produces exactly the same logits after reloading.
    reloaded = torch.load(output / 'checkpoint.pt', weights_only=True, map_location='cpu')
    restored = TinyDecoder(**reloaded['config'])
    restored.load_state_dict(reloaded['model_state'])
    model.eval()
    restored.eval()
    with torch.no_grad():
        x, _ = batches['validation'][0]
        torch.testing.assert_close(model(x)[0], restored(x)[0], atol=0, rtol=0)
    # Include continuations and a question; the question probes a known capability limitation.
    prompts = ['ROMEO:', 'JULIET:', 'The king', 'HAMLET:\n', 'To be, or not to be', 'What is your name?']
    # Store decoding settings as well as training settings to make output comparisons meaningful.
    report = dict(python=platform.python_version(), torch=torch.__version__, device='cpu', threads=1, dataset_sha256=dataset_hash, source_checkpoint=str(path.relative_to(ROOT)), source_sha256=original_hash, training_seconds=duration, additional_steps=5000, config=saved['config'], batch_size=8, learning_rate=saved['learning_rate'], weight_decay=optimizer.param_groups[0]['weight_decay'], reload_exact_match=True, evaluation_seed=123, common_targets=1024, prompts=prompts, generation=dict(seeds=[42,43,44], temperature=0.8, top_p=0.9, max_new_bytes=100, seed_policy='reset per prompt and seed for paired comparisons'), results=[])
    # Evaluate both endpoints of the experiment with one common procedure.
    for item in [saved, reloaded]:
        candidate = TinyDecoder(**item['config'])
        candidate.load_state_dict(item['model_state'])
        candidate.eval()
        with torch.no_grad():
            last_losses = []
            for ends in common:
                x, y = batch(val_data, ends)
                # Compare the last byte of each window, not averages over different targets.
                last_losses.append(F.cross_entropy(candidate(x)[0][:,-1,:], y[:,-1]).item())
        result = dict(step=item['step'], losses={name:evaluate(candidate, values) for name, values in batches.items()}, common_final_byte_validation_loss=sum(last_losses)/len(last_losses), outputs=[])
        if item['step'] == 5000:
            assert abs(result['common_final_byte_validation_loss'] - prior['results'][0]['common_final_byte_validation_loss']) < 1e-6
        # Six prompts x (one greedy + three sampled outputs) = 24 outputs per checkpoint.
        for prompt in prompts:
            for mode, seeds in [('greedy',[42]), ('sampled',[42,43,44])]:
                for seed in seeds:
                    # Unlike older comparisons, reset per prompt so prompt order cannot change sampling.
                    torch.manual_seed(seed)
                    text = generate(candidate,prompt,100,0.8,0.9,mode=='greedy')
                    result['outputs'].append(dict(prompt=prompt, mode=mode, seed=seed, continuation=text))
        report['results'].append(result)
        print(f'Evaluated step {item["step"]}: same-target loss={result["common_final_byte_validation_loss"]:.6f}; {len(result["outputs"])} continuations', flush=True)
    # Verify that the source checkpoint still has exactly the same file contents.
    assert original_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('All checks passed. Results:', output / 'results.json', flush=True)


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == '__main__':
    main()
