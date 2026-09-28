# Research on AI Chatbots

A hands-on Python and PyTorch research project exploring tokenization, embeddings, causal attention, Transformer training, and text generation.

**Last updated:** September 27, 2026  
**Working branch:** develop

See [Progess.md](Progess.md) for the full roadmap, experiment settings, loss history, limitations, and the preserved original report.

## Current progress

| Component | Verified result |
|---|---|
| PyTorch training loop | Linear model learned weight approximately 3 and bias approximately 2. |
| Byte-level BPE | All six test groups passed, including Unicode round trips and save/load. |
| Library comparison | Custom BPE and Hugging Face Tokenizers both used 276-entry vocabularies and passed sample round trips. |
| Embeddings | Token lookup, learned/sinusoidal positions, and gradient checks passed. |
| Attention | Custom single-head and multi-head causal and gradient checks passed. |
| Transformer decoder | Memorized a tiny sequence with loss 0.001536 and 100% training-token accuracy. |
| Language-model training | Trained on Tiny Shakespeare, resumed from 200 to 1,000 steps, and verified checkpoint reload. |
| Terminal generation | Loaded saved weights and compared greedy with temperature/top-p decoding. |

The nn.MultiheadAttention reference comparison remains unconfirmed. Retrieval, conversation memory, API/Docker work, and deployment are pending.

## Research findings

### Token coverage and learned merges

Our custom tokenizer starts with 256 byte tokens and learns 20 merges. Both tokenizers reconstructed punctuation, accented text, emoji, and empty input correctly.

| Text | UTF-8 bytes | Custom BPE tokens | Library BPE tokens |
|---|---:|---:|---:|
| hello world | 11 | 4 | 4 |
| Hello? | 6 | 5 | 5 |
| café 🤖 | 10 | 10 | 10 |
| Empty text | 0 | 0 | 0 |

Token IDs are tokenizer-specific. Matching counts on these samples do not imply identical vocabularies or behavior on every input. The separate code_tokenize project was reviewed as a syntax-analysis reference, not adopted as our BPE tokenizer.

### Learning and generalization are different from memorization

The small decoder reproduced hello world on four lines exactly. That was an overfitting check. The larger experiment used held-out text to measure next-byte prediction:

| Step | Training loss | Validation loss |
|---|---:|---:|
| 0 | 5.7875 | 5.7791 |
| 100 | 3.4313 | 3.4336 |
| 200 | 3.0636 | 3.0554 |
| 1,000 | 2.5064 | 2.4755 |

Both losses improved, without an obvious overfitting gap in the fixed evaluation batches. This does not establish fluent generation or performance on a final independent test set.

### Sampling changes outputs without changing model knowledge

With the same step-1,000 checkpoint and ROMEO prompt, greedy decoding repeated the, while temperature 0.8 and top-p 0.9 produced more varied but mostly malformed text. Sampling reduced repetition in this example; it did not make the model a reliable assistant.

## Baseline configuration

- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
- Contiguous 90/10 text split: 1,003,854 training bytes and 111,540 validation bytes.
- Tokenizer: fixed 256-byte vocabulary. The custom BPE tokenizer is a separate experiment and is not used in this training run.
- CPU; context 64 bytes; batch size 8; embedding size 64; four attention heads; two decoder blocks.
- AdamW; learning rate 0.0003; gradient norm clipped to 1.0.
- Evaluation every 100 steps on 10 fixed batches per split, without weight updates.
- Checkpoints save model, optimizer, configuration, completed step, and RNG state. Reload checks passed.

## Local implementation layout

The newer lesson files exist locally but were untracked when this documentation was updated. This documentation-only publication does not include those sources, the dataset, or checkpoints. Commands below describe the current local checkout, not a complete clean-clone installation.

```text
lessons/
  01_pytorch/exercises.py
  02_tokenizer/bpe.py
  02_tokenizer/compare_tokenizers.py
  03_embeddings/embeddings.py
  04_attention/single_head.py
  04_attention/multi_head.py
  05_decoder/decoder.py
  06_training/dataset.py
  06_training/train.py
  06_training/data/input.txt
  06_training/runs/<run>/
    losses.csv
    sample_<step>.txt
    checkpoint.pt
  07_chat/chat.py
```

The root minimal_chatbot.py is an earlier character-level demonstration that retrains on startup. Use the staged scripts for the newer checkpoint-based experiment.

## Environment and local commands

Recorded environment: Python 3.11, PyTorch 2.14.0+cpu, Hugging Face Tokenizers 0.23.2. Dependencies are not yet pinned. Use the same virtual-environment interpreter for installation and execution.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install tokenizers numpy
```

Skip environment creation if already configured. NumPy was missing in the reported runs; its warning did not prevent the checks from passing.

Run the local exercises:

```powershell
.\.venv\Scripts\python.exe lessons/02_tokenizer/bpe.py
.\.venv\Scripts\python.exe lessons/02_tokenizer/compare_tokenizers.py
.\.venv\Scripts\python.exe lessons/03_embeddings/embeddings.py
.\.venv\Scripts\python.exe lessons/04_attention/single_head.py
.\.venv\Scripts\python.exe lessons/04_attention/multi_head.py
.\.venv\Scripts\python.exe lessons/05_decoder/decoder.py
.\.venv\Scripts\python.exe lessons/06_training/dataset.py
```

Start a fresh 200-step training run:

```powershell
.\.venv\Scripts\python.exe lessons/06_training/train.py --steps 200
```

Use the recorded local checkpoint for generation:

```powershell
$checkpoint = ".\lessons\06_training\runs\20260927_204339_675021\checkpoint.pt"
.\.venv\Scripts\python.exe lessons/07_chat/chat.py --checkpoint "$checkpoint" --temperature 0.8 --top-p 0.9
```

For the comparison, exit with /quit and run:

```powershell
.\.venv\Scripts\python.exe lessons/07_chat/chat.py --checkpoint "$checkpoint" --greedy
```

Enter the same prompt, such as ROMEO, in both modes. Each prompt is independent. Generation uses the last 64 byte tokens and stops at its length limit. There is no trained end-of-response token, conversation memory, or instruction-following training.

To continue training, choose a total target greater than the checkpoint's completed step count. For example, this resumes the 1,000-step checkpoint for 1,000 additional updates:

```powershell
.\.venv\Scripts\python.exe lessons/06_training/train.py --resume "$checkpoint" --steps 2000
```

That extension is an example command; it has not been reported as executed.

## Next milestone

Build and evaluate document retrieval independently in Stage 8. The current generator is weak and has a short context window, so retrieving relevant passages and producing grounded answers must be evaluated separately. See the progress report for remaining verification and later API/deployment work.
