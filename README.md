# Research on AI Chatbots

A hands-on Python and PyTorch research project exploring tokenization, embeddings, causal attention, Transformer training, and text generation.

**Last updated:** October 2, 2026
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
| Language-model training | Resumed Tiny Shakespeare training to 3,000 total steps; validation loss 2.2393 and checkpoint reload passed. |
| Terminal generation | Loaded saved weights and compared greedy with temperature/top-p decoding. |
| Document retrieval | Loaded two documents, created 79 chunks, and indexed 384-dimensional embeddings with FAISS; preparation and index checks passed. |
| RAG answer generation | Not verified. Retrieval returns passages; the cloud API test failed with HTTP 401. |

The nn.MultiheadAttention reference comparison remains unconfirmed. Grounded answer generation, conversation memory, a serving API, Docker, and deployment remain pending.

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
| 2,000 | 2.3546 | 2.3422 |
| 3,000 | 2.2426 | 2.2393 |

Both losses improved, without an obvious overfitting gap in the fixed evaluation batches. This does not establish fluent generation or performance on a final independent test set.

### Sampling changes outputs without changing model knowledge

With the same step-1,000 checkpoint and ROMEO prompt, greedy decoding repeated the, while temperature 0.8 and top-p 0.9 produced more varied but mostly malformed text. Sampling reduced repetition in this example; it did not make the model a reliable assistant.

At 3,000 steps, greedy samples still repeated "the". A sampled ROMEO continuation contained invented words such as "Burd mat yom" alongside English words. This is English-like gibberish, not a coherent response in another language. Lower next-byte prediction loss has not yet produced fluent text or an instruction-following chatbot.

### Retrieval finds evidence but does not write answers

The document experiment uses 500-character chunks with 100-character overlap, sentence-transformers/all-MiniLM-L6-v2 embeddings, and a FAISS similarity index. In the recorded snapshot, README.md and Progess.md produced 79 chunks.

For "How many tokens were in our BPE vocabulary?", the passage explicitly stating **276 = 256 base bytes + 20 merges** ranked fourth (similarity 0.5137). Top-three retrieval missed it; top-five included it. The language-model training vocabulary is separately fixed at 256 bytes. An unrelated weather question still returned irrelevant passages, so similarity scores alone do not establish answerability.

Connecting an answer model remains unfinished: the OpenRouter test returned HTTP 401, and no successful generated RAG answer has been recorded. The document copies used by retrieval are snapshots and must be refreshed separately when root documentation changes.

## Baseline configuration

- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
- Contiguous 90/10 text split: 1,003,854 training bytes and 111,540 validation bytes.
- Tokenizer: fixed 256-byte vocabulary. The custom BPE tokenizer is a separate experiment and is not used in this training run.
- CPU; context 64 bytes; batch size 8; embedding size 64; four attention heads; two decoder blocks.
- AdamW; learning rate 0.0003; gradient norm clipped to 1.0.
- Evaluation every 100 steps on 10 fixed batches per split, without weight updates.
- Checkpoints save model, optimizer, configuration, completed step, and RNG state. Reload checks passed.

## Local implementation layout

Lessons through Stage 7 and earlier training artifacts were published to develop. The October 2 training run and Stage 8 files are local additions not yet committed at the time of this update. Commands below describe the local checkout.

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
  08_rag/retrieve.py
  08_rag/semantic_search.py
  08_rag/test_api.py
  08_rag/documents/
```

The root minimal_chatbot.py is an earlier character-level demonstration that retrains on startup. Use the staged scripts for the newer checkpoint-based experiment.

## Environment and local commands

Recorded environment: Python 3.11, PyTorch 2.14.0+cpu, Hugging Face Tokenizers 0.23.2. Dependencies are not yet pinned. Use the same virtual-environment interpreter for installation and execution.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install tokenizers numpy
```

Skip environment creation if already configured. NumPy was subsequently installed with the retrieval dependencies; its earlier missing-package warning did not prevent training checks from passing.

Retrieval dependencies can be installed in the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install sentence-transformers faiss-cpu numpy requests
```

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
$checkpoint = ".\lessons\06_training\runs\20261002_123352_952051\checkpoint.pt"
.\.venv\Scripts\python.exe lessons/07_chat/chat.py --checkpoint "$checkpoint" --temperature 0.8 --top-p 0.9
```

For the comparison, exit with /quit and run:

```powershell
.\.venv\Scripts\python.exe lessons/07_chat/chat.py --checkpoint "$checkpoint" --greedy
```

Enter the same prompt, such as ROMEO, in both modes. Each prompt is independent. Generation uses the last 64 byte tokens and stops at its length limit. There is no trained end-of-response token, conversation memory, or instruction-following training.

The completed October 2 experiment resumed the 1,000-step checkpoint for 2,000 additional updates. The steps argument is the total target, not the number of extra updates:

```powershell
$previousCheckpoint = ".\lessons\06_training\runs\20260927_204339_675021\checkpoint.pt"
.\.venv\Scripts\python.exe lessons/06_training/train.py --resume "$previousCheckpoint" --steps 3000
```

Run the document preparation checks and passage search:

```powershell
.\.venv\Scripts\python.exe lessons/08_rag/retrieve.py
.\.venv\Scripts\python.exe lessons/08_rag/semantic_search.py
```

Search prints source passages, not generated answers. Type /quit to exit. The embedding model may need downloading on first use. Keep API keys out of source files, documentation, and Git.

## Next milestone

Inspect the model and generation code before committing to longer training, and compare decoding modes using fixed prompts and settings. For Stage 8, evaluate retrieval against known answers and unrelated questions, then connect and test an answer model with citations and an insufficient-evidence response. The current 64-byte generator has not demonstrated the ability to answer from retrieved passages. Docker and deployment follow after the application works end to end.
