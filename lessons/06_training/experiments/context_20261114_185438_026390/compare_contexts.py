"""Fresh, matched context-length experiment; preserves earlier training runs."""
import csv
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from train import TinyDecoder, DEFAULT_CONFIG, evaluate, generate_sample
from dataset import load_data, get_batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'lessons/07_chat'))
from chat import generate


def main():
    torch.set_num_threads(1)
    train_data, val_data = load_data()
    output = Path(__file__).parent / 'experiments' / datetime.now(timezone.utc).strftime('context_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the exact implementation used for this experiment.
    sources = [Path(__file__), Path(__file__).with_name('train.py'), Path(__file__).with_name('dataset.py'), ROOT / 'lessons/05_decoder/decoder.py', ROOT / 'lessons/04_attention/multi_head.py', ROOT / 'lessons/07_chat/chat.py']
    for source in sources:
        (output / source.name).write_bytes(source.read_bytes())
    metadata = dict(python=platform.python_version(), torch=torch.__version__, device='cpu', threads=1, steps=5000, batch_size=8, learning_rate=0.0003, weight_decay=0.01, seed=42, data_seed=456, evaluation_seed=123, dataset_sha256=hashlib.sha256((ROOT / 'lessons/06_training/data/input.txt').read_bytes()).hexdigest(), design='Same initial shared weights and first 64 positions. Same sampled target endpoints. Longer context sees twice as many training tokens per update. Equal steps, not equal compute/token budget.', results=[])
    (output / 'results.json').write_text(json.dumps(metadata, indent=2))
    print('Experiment:', output, flush=True)

    torch.manual_seed(42)
    initial = TinyDecoder(**DEFAULT_CONFIG).state_dict()
    torch.manual_seed(123)
    eval_endpoints = {name: torch.randint(128, len(data), (10, 8)) for name, data in [('train', train_data), ('validation', val_data)]}
    # A separate, larger set compares identical final target bytes, giving each model its full context.
    common_endpoints = torch.randint(128, len(val_data), (128, 8))

    def batch(data, endpoints, length):
        return (torch.stack([data[e-length:e] for e in endpoints.tolist()]), torch.stack([data[e-length+1:e+1] for e in endpoints.tolist()]))

    def new_model(length):
        torch.manual_seed(42)
        model = TinyDecoder(**dict(DEFAULT_CONFIG, max_length=length))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name == 'position_embedding.weight':
                    parameter[:64].copy_(initial[name])
                else:
                    parameter.copy_(initial[name])
        return model

    for length in (64, 128):
        model = new_model(length)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        torch.manual_seed(456)
        start = time.perf_counter()
        for _ in range(100):
            endpoints = torch.randint(128, len(train_data), (8,))
            x, y = batch(train_data, endpoints, length)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        seconds = time.perf_counter() - start
        metadata.setdefault('benchmark', {})[str(length)] = {'100_steps_seconds': seconds, 'estimated_5000_training_seconds': seconds * 50}
        print(f'Benchmark context={length}: 100 updates {seconds:.2f}s; estimated training {seconds*50:.1f}s', flush=True)
    (output / 'results.json').write_text(json.dumps(metadata, indent=2))

    for length in (64, 128):
        folder = output / str(length)
        folder.mkdir()
        model = new_model(length)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        evaluation = {name: [batch(data, ends, length) for ends in eval_endpoints[name]] for name, data in [('train', train_data), ('validation', val_data)]}
        torch.manual_seed(456)
        start = time.perf_counter()
        with (folder / 'losses.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['step', 'train_loss', 'validation_loss'])
            for step in range(5001):
                if step:
                    model.train()
                    ends = torch.randint(128, len(train_data), (8,))
                    x, y = batch(train_data, ends, length)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(x, y)[1]
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                if step % 500 == 0:
                    losses = {name: evaluate(model, batches) for name, batches in evaluation.items()}
                    writer.writerow([step, losses['train'], losses['validation']])
                    handle.flush()
                    sample = generate_sample(model)
                    (folder / f'sample_{step:04d}.txt').write_text(sample)
                    saved = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), config=dict(DEFAULT_CONFIG, max_length=length), step=step, batch_size=8, learning_rate=0.0003, tokenizer='utf8_bytes_256', rng_state=torch.get_rng_state(), resumed_from=None)
                    torch.save(saved, folder / 'checkpoint.tmp')
                    (folder / 'checkpoint.tmp').replace(folder / 'checkpoint.pt')
                    print(f'context={length} step={step} train={losses["train"]:.4f} validation={losses["validation"]:.4f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        duration = time.perf_counter() - start
        saved = torch.load(folder / 'checkpoint.pt', weights_only=True, map_location='cpu')
        restored = TinyDecoder(**saved['config'])
        restored.load_state_dict(saved['model_state'], strict=True)
        restored.eval()
        model.eval()
        with torch.no_grad():
            x, _ = evaluation['validation'][0]
            torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)
            common_losses = []
            for ends in common_endpoints:
                x, y = batch(val_data, ends, length)
                logits, _ = restored(x)
                common_losses.append(F.cross_entropy(logits[:, -1, :], y[:, -1]).item())
        result = dict(context=length, parameters=sum(p.numel() for p in model.parameters()), seconds=duration, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_memory_note='macOS cumulative process high-water mark, including benchmark and previous runs; not isolated model memory', losses=losses, common_final_byte_validation_loss=sum(common_losses)/len(common_losses), common_targets=1024, reload_exact_match=True, training_tokens=5000*8*length, outputs={})
        for mode in ('greedy', 'sampled'):
            torch.manual_seed(42)
            result['outputs'][mode] = [{'prompt': prompt, 'continuation': generate(restored, prompt, 100, 0.8, 0.9, mode=='greedy')} for prompt in ['ROMEO:', 'JULIET:', 'The king']]
        result['generation_settings'] = dict(seed=42, temperature=0.8, top_p=0.9, max_new_bytes=100, seed_policy='reset once per mode')
        metadata['results'].append(result)
        (output / 'results.json').write_text(json.dumps(metadata, indent=2))
        print('Completed:', json.dumps(result), flush=True)
    print('All training and reload checks passed. Results:', output / 'results.json', flush=True)


if __name__ == '__main__':
    main()
