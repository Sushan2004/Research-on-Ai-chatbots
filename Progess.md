# Chatbot Research Project — Roadmap and Progress Report

**Updated:** October 2, 2026
**Branch:** develop  
**Repository:** https://github.com/Sushan2004/Research-on-Ai-chatbots

## October 2 progress report

This update records the learner's terminal results and the latest saved loss log. Training and API requests were not rerun for this documentation update. Earlier reports below are historical; this section supersedes their status labels.

### Current progress

| Stage | Status |
|---|---|
| 1–5. Fundamentals through decoder | Previously recorded exercises and tiny-sequence overfitting passed; separate tensor/autograd run confirmations and the PyTorch attention reference comparison remain unrecorded. |
| 6. Language-model training | Reached 3,000 total optimizer steps; checkpoint reload passed. |
| 7. Terminal generation | Greedy and temperature/top-p inference work, but output remains repetitive or malformed. No conversation memory or instruction training. |
| 8. RAG | Document preparation and semantic retrieval tested. Answer generation and answerability handling remain incomplete. |
| 9. Docker and serving API | Not completed. The external API connectivity test is not a deployed serving API. |
| 10. Deployment | Not completed. |

### Extended training experiment

Resumed `20260927_204339_675021/checkpoint.pt` at step 1,000 and trained to step 3,000: **2,000 additional optimizer updates**. The `--steps` argument specifies the total target. Results are saved under `lessons/06_training/runs/20261002_123352_952051/`.

The baseline remains a CPU-trained, two-block decoder with 64-dimensional embeddings, four attention heads, a 64-byte context, and batch size 8. Training uses Tiny Shakespeare, a fixed 256-byte vocabulary, AdamW at 0.0003, and gradient clipping at 1.0. The separate 276-entry BPE experiment is not the tokenizer for this model.

| Total step | Training loss | Validation loss |
|---|---:|---:|
| 1,000 | 2.5064 | 2.4755 |
| 1,500 | 2.4252 | 2.4116 |
| 2,000 | 2.3546 | 2.3422 |
| 2,500 | 2.3059 | 2.2920 |
| 3,000 | 2.2426 | 2.2393 |

Checkpoint reload checks passed. Training and validation losses declined with no large gap on the recorded evaluation batches. This measures improved next-byte prediction; it does not prove fluent generation or performance on an independent final test set.

### Generation findings

Greedy output at step 3,000 still repeatedly generated "the". A sampled continuation of `ROMEO` began:

```text
RETIUCK:
Burd mat yom you mouler blal the car me he the sucarouly ncat,
```

This output imitates English/Shakespeare-like spelling and formatting but is mostly gibberish. Sampling changes which tokens are selected; it does not supply missing language competence. The model has not become a reliable conversational assistant. Inspect training, causal masking, context handling, and generation code before assuming more steps alone will resolve the problem; no implementation bug has yet been established.

### Retrieval experiment

- Loaded document snapshots: Progess.md (24,135 characters) and README.md (7,075 characters).
- Split them into 79 chunks of up to 500 characters with 100-character overlap, retaining source names and character offsets.
- Document loading, chunking, and preparation checks passed.
- Encoded passages with sentence-transformers/all-MiniLM-L6-v2 into normalized 384-dimensional vectors and indexed them with FAISS. Index checks passed.

For the question "How many tokens were in our BPE vocabulary?", the explicit answer **276** appeared at rank 4 in Progess.md, chunk 5, with similarity 0.5137. The top three passages discussed related token counts or the separate 256-byte training vocabulary. Increasing retrieval to five passages exposed the relevant evidence, but did not fix its ranking.

The unrelated question "What is the weather tomorrow?" returned irrelevant passages with scores around 0.12–0.14. Search always returning neighbors is not evidence that the documents contain an answer. No calibrated rejection rule or validated answer-generation fallback has been demonstrated.

These character counts and rankings refer to the tested document snapshot, not the newly updated root files. Copies under `lessons/08_rag/documents/` are unchanged by this documentation update and will need a separate refresh and reindex.

### Answer-model integration and environment

Retrieval currently prints passages rather than a synthesized answer. A local answer model was discussed but no completed local-model integration was verified. The OpenRouter API test returned **HTTP 401**; successful authentication and grounded answer generation remain unverified. No API keys belong in this report or the repository.

NumPy and retrieval dependencies were installed. Reported versions include NumPy 2.4.6, sentence-transformers 6.1.0, faiss-cpu 1.15.1, and transformers 5.18.0. Dependencies still need a reproducible pinned environment. The earlier missing-NumPy warning is historical.

### Repository status and next work

Lessons through Stage 7 and the earlier artifacts were published to develop. At the time of this update, the October 2 training run and Stage 8 directory are local, untracked additions. This update changes only the root README.md and Progess.md; it does not publish them or alter lesson files.

1. Review the model and generation implementation, then compare greedy and sampling outputs with fixed prompts and recorded settings.
2. Build a small retrieval evaluation set with known answers, including BPE vocabulary versus training vocabulary and unrelated questions.
3. Connect an answer model only after a successful standalone test, then check source citations and insufficient-evidence behavior.
4. Complete the application before moving to Docker, a serving API, and deployment.

## Historical September 27 progress report

This update records results reported by the learner in terminal output. Source files were inspected for documentation consistency; the experiments were not rerun for this report. The September 25 roadmap is preserved below as historical context. Its old status labels are superseded by this update.

### Milestone status

| Stage | Current evidence and status |
|---|---|
| 1. PyTorch fundamentals | Linear training passed previously; separate tensor/autograd terminal confirmations remain unrecorded. |
| 2. Tokenizer | Custom byte-level BPE and library comparison passed. |
| 3. Embeddings and positions | Token lookup, learned positions, sinusoidal positions, and gradient checks passed. |
| 4. Attention | Single-head and custom multi-head shape, causal, and gradient checks passed. PyTorch reference comparison remains unconfirmed; compare_attention.py was not present during this update. |
| 5. Decoder | Tiny-sequence overfitting and generation passed. |
| 6. Language-model training | Dataset checks, 200-step baseline, resume to 1,000 steps, and checkpoint reload passed. |
| 7. Terminal generation | Checkpoint loading, greedy generation, temperature/top-p sampling, and /quit tested. Independent prompts only; conversation memory and dialogue training are not implemented. |
| 8. Retrieval | Not started. Next proposed experiment is retrieval evaluation independent of this limited generator. |
| 9. API and Docker | Not started. |
| 10. Deployment | Not started. |

### 1. Tokenizer implementation and comparison

Implemented encode_bytes, decode_bytes, count_pairs, merge_pair, train_bpe, encode_bpe, decode_bpe, save_tokenizer, and load_tokenizer. All six groups of checks passed: byte encoding, pair counting, pair merging, BPE training, encoding/decoding, and save/load.

The custom tokenizer learned 20 merges on top of 256 base byte tokens, producing a vocabulary of 276. Encoding applies learned rules without changing them. Decoding expands merged tokens into byte sequences before UTF-8 decoding.

Compared with Hugging Face Tokenizers 0.23.2 using the same training text and vocabulary budget, with prefix-space insertion and regex splitting disabled:

| Input | UTF-8 bytes | Custom BPE tokens | Library BPE tokens |
|---|---:|---:|---:|
| hello world | 11 | 4 | 4 |
| Hello? | 6 | 5 | 5 |
| café 🤖 | 10 | 10 | 10 |
| Empty string | 0 | 0 | 0 |

Both tokenizers reconstructed all examples exactly. Different token IDs are expected because vocabulary mappings differ. Matching counts on these examples do not prove universal equivalence. The unchanged byte count for café 🤖 shows that the learned rules did not merge its adjacent byte tokens.

Reviewed code_tokenize as a separate research reference: it produces syntax-aware program token objects from syntax trees rather than learning byte-pair vocabulary IDs. It was not integrated into the project.

### 2. Embeddings and positional information

Verified a token table with shape [10, 4], containing 40 parameters. Input [2, 3] became output [2, 3, 4]. Repeated token IDs retrieved identical vectors.

Added an [8, 4] learned position table containing 32 parameters, plus an alternative fixed sinusoidal encoding. Position vectors distinguish occurrences of the same token at different positions. Gradient checks confirmed updates can reach used learned-table rows; unused rows received zero gradients in the probe calculation. These were artificial gradient checks, not semantic embedding training.

### 3. Causal single-head and multi-head attention

Single-head attention passed with output [1, 3, 2]. Custom multi-head attention passed with input/output [2, 5, 8], two heads, and four output features per head.

Each head receives all eight input features and uses its own Q/K/V projections. A causal mask permits zero-based key positions less than or equal to the query position. Changing the final input did not alter earlier outputs. Gradient checks passed.

A weight-matched comparison with nn.MultiheadAttention was supplied as a follow-up, but its execution is not confirmed. Do not report numerical equivalence to PyTorch as tested yet.

### 4. Minimal decoder overfitting

Assembled token and learned position embeddings, two decoder blocks, pre-layer normalization, residual connections, feedforward networks, final normalization, and vocabulary projection. Inputs and targets were shifted by one token for next-token prediction.

| Measurement | Result |
|---|---:|
| Final training loss | 0.001536 |
| Training-token accuracy | 100.00% |
| Generated text | hello world repeated on four lines |

The exact-generation and overfitting checks passed. This establishes successful memorization of the tiny example, not general language understanding.

### 5. Dataset and training setup

Dataset: [Tiny Shakespeare from karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).

The text was split contiguously into 90% training and 10% validation before byte encoding and window sampling. Training contained 1,003,854 bytes; validation contained 111,540 bytes. Dataset checks verified [4, 32] input/target batches, one-token shifts, valid byte IDs, and window boundaries.

The larger training run deliberately uses the fixed 256-byte vocabulary, not the learned 276-token BPE vocabulary. Keep this distinction when reproducing the experiment.

| Setting | Value |
|---|---|
| Device | CPU |
| Vocabulary | 256 UTF-8 byte IDs |
| Context length | 64 byte tokens |
| Batch size | 8 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Decoder blocks | 2 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Gradient clipping | Maximum norm 1.0 |
| Evaluation | Every 100 steps; 10 fixed batches per split |
| Training seed | 42 |
| Evaluation batch seed | 123 |

Validation runs used evaluation mode with gradient tracking disabled and did not update weights. The initial run completed 200 optimizer steps. A continuation restored weights, optimizer state, configuration, and random-number-generator state, then performed 800 additional updates to reach step 1,000.

### 6. Recorded loss history

| Step | Training loss | Validation loss |
|---|---:|---:|
| 0 | 5.7875 | 5.7791 |
| 100 | 3.4313 | 3.4336 |
| 200 | 3.0636 | 3.0554 |
| 300 | 2.8477 | 2.8193 |
| 400 | 2.7264 | 2.6969 |
| 500 | 2.6504 | 2.6172 |
| 600 | 2.6012 | 2.5714 |
| 700 | 2.5689 | 2.5387 |
| 800 | 2.5398 | 2.5156 |
| 900 | 2.5284 | 2.4949 |
| 1,000 | 2.5064 | 2.4755 |

Loss declined on both the training and held-out batches. There is no obvious overfitting gap in these measurements, but this is a small fixed validation sample and not an independent final test set.

Greedy samples progressed from random bytes to whitespace and recognizable but repetitive fragments. Validation loss measures prediction given real preceding text, whereas free generation conditions on the model's own outputs. Lower validation loss therefore does not guarantee coherent generated text.

Checkpoint reload checks passed after both runs. The final local checkpoint is:

```text
lessons/06_training/runs/20260927_204339_675021/checkpoint.pt
```

Each run also records losses.csv and numbered text samples. Resuming to --steps 1000 means 1,000 total steps, not 1,000 additional steps. The dataset and decoder implementation were kept unchanged for the continuation.

### 7. Terminal generation and decoding experiment

Loaded the step-1,000 checkpoint without retraining. Compared the same ROMEO prompt with greedy decoding and temperature 0.8 / top-p 0.9 sampling.

| Mode | Observation |
|---|---|
| Greedy | Began RNTh and repeatedly generated the. |
| Temperature/top-p | More varied letter and word fragments, but mostly malformed text. |

Sampling reduced repetition in this example. It did not demonstrate fluent language, factual answers, or instruction following. The comparison used one prompt and one sampled continuation, so it is not a broad quality benchmark.

The interface accepts independent prompts, retains only the last 64 byte tokens as generation context, stops at its generation-length limit, and exits on /quit. It has no persistent conversation history or trained end-of-response token. Invalid generated UTF-8 is displayed with replacement characters.

### Limitations and remaining work

- The legacy minimal_chatbot.py remains separate from the staged checkpoint-based implementation. Its earlier limitations must not be confused with the newer scripts.
- The BPE tokenizer is tested independently but is not connected to the current language-model training run.
- Generated language remains weak after 1,000 steps; a terminal interface does not establish chatbot reasoning or answering ability.
- GPU execution, full PyTorch attention equivalence, and a final independent evaluation remain unverified.
- Dependencies are not pinned. Recorded environment: Python 3.11, torch 2.14.0+cpu, tokenizers 0.23.2. NumPy was still missing in reported runs, causing a non-fatal warning.
- New lesson sources and generated artifacts were still untracked at documentation review. This publication covers README.md and Progess.md only; it does not publish checkpoints, datasets, or lesson code.

### Next actions

1. Preserve the baseline logs and keep dataset/model settings attached to future experiments.
2. Close the unconfirmed attention-reference comparison when revisiting validation.
3. Begin Stage 8 by building a small document index and checking retrieved passages against known questions.
4. Evaluate retrieval independently before asking this weak, short-context model to produce grounded answers.
5. Keep conversation memory, API/container work, and deployment marked pending until implemented and tested.

---

## Historical roadmap — September 25, 2026

The original report below is retained for the plan and earlier results. Use the September 27 status table above for current progress.

# Chatbot Research Project — Roadmap and Progress Report

**Report date:** September 25, 2026  
**Project:** Building and understanding a chatbot from scratch  
**Repository:** [Research-on-Ai-chatbots](https://github.com/Sushan2004/Research-on-Ai-chatbots)  
**Working branch:** `develop`

**Local project folder:**
```text
C:\Users\Sushan Adhikari\Desktop\one drive\OneDrive - Missouri State University\codex Project\Chatbox
```

This report is based on the code and terminal outputs shared in our conversation. No commands were executed or files changed to prepare it.

## 1. Project objective

Build a small chatbot while learning how each component works. We will begin with PyTorch fundamentals, implement the main language-model components ourselves, train a tiny model, and gradually add conversation handling, retrieval, an API, and deployment.

The initial goal is **an understandable, testable learning system**. A tiny model trained on a small corpus will have limited language and question-answering capabilities. We will measure those limits and distinguish infrastructure improvements from improvements in the model itself.

## 2. How we will work

For each stage:

1. Explain the concept briefly.
2. Provide small exercises with TODOs.
3. You implement and run the code.
4. Review your attempt and debugging output.
5. Show solutions after you attempt the exercise.
6. Check understanding and record results.
7. Advance only after you confirm the stage is built and tested.

**Execution preference:** I will not execute commands without your approval. By default, I will provide commands for you to run.

You may use Gemini for deeper research and Claude for architecture feedback while we focus on implementation and experiments.

## 3. Current progress

| Area | Status | Evidence |
|---|---|---|
| Repository connection | Complete | Local checkout tracks `origin/develop` |
| Move to new project folder | Complete | Repository and lessons copied to the requested Chatbox folder |
| Python virtual environment | Working | You successfully ran training through `.venv` |
| PyTorch installation | Working | Reported version: `2.14.0+cpu` |
| Tensor exercise | Implemented; test confirmation pending | Shared code contains tensor creation, matrix multiplication, and checks |
| Autograd exercise | Implemented; test confirmation pending | Shared code calculates the polynomial and calls `backward()` |
| `nn.Module` exercise | Verified through training | `LineModel` uses `nn.Linear(1, 1)` |
| Training loop | Passed | Loss decreased and final assertions passed |
| CPU execution | Verified | Output reported `Training checks passed on cpu` |
| GPU execution | Not tested | Current installation is a CPU build |
| Transformer demonstration | Ran successfully, with limitations | Generated text through the terminal chat loop |
| Unsupported-character handling | Fix provided; verification pending | Original program crashed on `?` |
| Later learning stages | Not completed | Running the demonstration does not replace the staged exercises |
| Git commits and push | Not confirmed | No commit or push was recorded in our conversation |

### Verified training result

Your linear model learned the target relationship:

```text
y = 3*x + 2
```

Recorded output:

```text
step=  0 loss=3.060877
step= 50 loss=0.001525
step=100 loss=0.000001
step=150 loss=0.000000
step=200 loss=0.000000

Training checks passed on cpu. f(2)=8.0000

Weight: 2.9999983310699463
Bias:   1.999999761581421
```

This confirms that your forward pass, loss calculation, backpropagation, and optimizer update worked together. The displayed zero loss is rounded.

## 4. Full learning and implementation roadmap

### Stage 1 — PyTorch fundamentals

**Status:** Mostly complete; final checks pending.

**Learn**
- Tensor shapes, dtypes, and broadcasting.
- Matrix multiplication.
- Autograd and accumulated gradients.
- Parameters and `nn.Module`.
- Loss functions and optimization.
- CPU/GPU device placement.
- Evaluation mode versus disabling gradient tracking.

**Build**
- Tensor operations exercise.
- Scalar differentiation exercise.
- A model that learns `y = 3*x + 2`.

**Completion criteria**
- Tensor and autograd checks pass.
- Training reaches the required loss and prediction checks.
- Explain `zero_grad()`, `backward()`, and `step()`.
- Explain why model parameters and inputs must share a device.
- Confirm readiness to proceed.

**Deliverable:** Completed fundamentals exercises and recorded results.

### Stage 2 — Build a tokenizer

**Status:** Not started in the learning sequence.

**Learn**
- Text, bytes, tokens, vocabulary, and token IDs.
- How BPE learns frequent adjacent-pair merges.
- Why encoding and decoding must be consistent.
- How unseen characters affect tokenization.

**Build**
- A simple byte-level BPE tokenizer using standard Python.
- Pair counting, merge selection, and merge application.
- Encoding and decoding.
- Saved vocabulary and merge rules.
- A comparison with `tiktoken` or Hugging Face Tokenizers.

**Tests**
- Exact encode/decode round trips.
- Empty text, punctuation, repeated text, and Unicode.
- Deterministic merge order.
- Save/load consistency.

**Research output:** Compare character, byte, and BPE token counts on sample text.

**Deliverable:** Tokenizer implementation, tests, and comparison notes.

### Stage 3 — Embeddings and positional information

**Status:** Not started; present in the demonstration script.

**Learn**
- Turning token IDs into learned vectors.
- Batch, sequence, and embedding dimensions.
- Why attention needs position information.
- Learned versus sinusoidal positional encodings.

**Build**
- Token embedding lookup.
- Sinusoidal positional encoding.
- Learned position embeddings.
- Combined token and position representations.

**Tests**
- Verify shapes: `[batch, sequence, embedding]`.
- Confirm embedding parameters receive gradients.
- Check position limits and device consistency.

**Research output:** Compare the properties of learned and sinusoidal positions.

**Deliverable:** Tested embedding and position modules.

### Stage 4 — Self-attention from scratch

**Status:** Not started; present in the demonstration script.

**Learn**
- Query, key, and value projections.
- Scaled dot-product attention.
- Softmax over attention scores.
- Causal masking.
- Splitting and combining multiple heads.

**Build**
- Manual single-head attention.
- Causal masking.
- Multi-head attention and output projection.
- Comparison with PyTorch attention after the manual implementation works.

**Tests**
- Verify intermediate tensor shapes.
- Check attention probabilities before dropout.
- Verify future-token changes cannot affect earlier outputs.
- Check gradients and compare against a reference implementation.

**Research output:** Inspect attention weights for a small sequence.

**Deliverable:** Tested attention modules and a shape walkthrough.

### Stage 5 — Assemble a minimal Transformer decoder

**Status:** Demonstration executed; independent implementation pending.

**Learn**
- Residual connections.
- Layer normalization.
- Feedforward networks.
- Decoder block composition.
- Vocabulary logits and shifted next-token targets.

**Build**
- Decoder block.
- Stacked decoder model.
- Output projection and cross-entropy loss.
- A tiny overfitting experiment.

**Tests**
- Forward output shape: `[batch, sequence, vocabulary]`.
- Finite loss and gradients.
- Causal behavior across the full model.
- Successful overfitting on a tiny fixed dataset.

**Deliverable:** A minimal decoder whose components you can explain and test.

### Stage 6 — Train a tiny language model

**Status:** Tiny demonstration trained; structured experiment pending.

**Learn**
- Dataset preparation and batching.
- Training versus validation loss.
- Overfitting and generalization.
- Learning rate, context length, and model size.
- Checkpoints and reproducibility.

**Build**
- Training pipeline for a small corpus.
- Separate training and validation data.
- Periodic loss measurement and text samples.
- Checkpoint saving, loading, and training resumption.

**Tests**
- Validation uses held-out data.
- Loss and generation are logged at fixed intervals.
- Reloaded checkpoints reproduce evaluation outputs.
- Sampling uses evaluation mode.

**Research output**
- Loss curves.
- Generated text at several checkpoints.
- One controlled comparison, such as two context lengths.

**Deliverable:** Saved model, configuration, training logs, and experiment report.

### Stage 7 — Wrap the model as a chatbot

**Status:** Basic demonstration exists; conversational behavior incomplete.

**Learn**
- Autoregressive inference.
- Temperature and top-p sampling.
- Conversation formatting and history.
- Context limits and stopping conditions.

**Build**
- Terminal interface.
- Consistent user/assistant formatting in data and prompts.
- Conversation history management.
- Safe handling of unsupported input.
- Exit/reset commands.
- Loading a saved model without retraining.

**Tests**
- Empty and unsupported input do not crash.
- Generation length is bounded.
- Prompt text is separated from generated output.
- History fits within the context window.
- Restarting the application loads the checkpoint.

**Research output:** Compare outputs across temperature and top-p settings.

**Deliverable:** A terminal chatbot with documented capability limits.

### Stage 8 — Add retrieval-augmented generation

**Status:** Not started.

**Learn**
- Document loading and chunking.
- Retrieval embeddings and similarity search.
- Retrieval quality versus answer quality.
- Building prompts from retrieved context.

**Build**
- A small document collection.
- FAISS or Chroma index.
- Retrieval pipeline.
- Context formatting and source references.
- Retrieval inspection tools.

**Tests**
- Known questions retrieve relevant passages.
- Irrelevant questions are identified.
- Retrieved context fits the model’s input budget.
- Compare answers with and without retrieval.

**Important design checkpoint:** Our tiny model may not reliably use retrieved passages. If that happens, we will demonstrate retrieval separately and compare with a more capable pretrained model, while keeping the scratch model as a learning artifact.

**Deliverable:** Retrieval pipeline and a grounded-answer evaluation report.

### Stage 9 — Containerize and expose an API

**Status:** Not started.

**Learn**
- Separating training from inference.
- HTTP request/response schemas.
- Dependency management.
- Model loading and process lifecycle.
- Docker images and persistent artifacts.

**Build**
- FastAPI inference service.
- Health endpoint.
- Validated generation request.
- Dependency file and Dockerfile.
- Documented model-checkpoint location.

**Tests**
- API starts with the expected checkpoint.
- Invalid requests return useful errors.
- Generation limits are enforced.
- Container can serve a sample request.

**Deliverable:** Containerized API and usage instructions.

### Stage 10 — Deploy and evaluate

**Status:** Not started.

**Learn**
- Local versus cloud deployment.
- CPU/GPU resource requirements.
- Startup time, memory, and inference latency.
- Logging, access controls, and cost considerations.

**Build**
- Local Docker deployment.
- End-to-end test procedure.
- Deployment configuration and troubleshooting guide.
- Cloud deployment plan; actual cloud deployment only with approval.

**Tests**
- Health and generation requests pass.
- Measure latency and memory usage.
- Confirm restart behavior and checkpoint availability.
- Verify documented setup on a clean environment.

**Deliverable:** Reproducible local deployment and a cloud-readiness report.

## 5. Findings from the current chatbot demonstration

The demonstration includes character tokenization, learned embeddings, causal multi-head attention, decoder blocks, training, and generation.

Its observed limitations are:

| Finding | Explanation | Planned action |
|---|---|---|
| Repeats training phrases | Very small corpus encourages memorization | Add held-out evaluation and larger data in Stage 6 |
| Crashes on `?` | Character absent from vocabulary | Verify the input guard; address coverage in Stage 2 |
| Continues prompts instead of reliably answering | Trained as a text continuation model | Add appropriate conversation data and formatting in Stage 7 |
| No conversation memory | Each prompt is processed independently | Add bounded history in Stage 7 |
| Retrains at startup | No checkpoint-loading path | Add persistence in Stage 6 |
| Dropout remains active during chat | Evaluation mode is not enabled | Add and verify `model.eval()` |
| Last valid batch window is skipped | Sampling upper bound is off by one | Correct and test the sampler |

These are learning findings, not evidence that the overall approach has failed.

## 6. Research records we will maintain

For each meaningful experiment, record:

| Field | What to capture |
|---|---|
| Question | What are we trying to learn? |
| Hypothesis | What result do we expect, and why? |
| Configuration | Dataset, tokenizer, model size, context, optimizer, seed |
| Change | Which variable changed from the baseline? |
| Measurements | Loss, generation samples, runtime, or retrieval quality |
| Interpretation | What does the evidence support? |
| Limitations | What has not been established? |
| Next step | What experiment follows from the result? |

We will avoid changing many settings at once when the goal is understanding cause and effect.

## 7. Immediate next actions

1. Confirm the tensor and autograd exercises pass.
2. Finish the Stage 1 understanding checks.
3. Record Stage 1 as complete after your confirmation.
4. Begin the BPE tokenizer exercise.
5. Keep the current Transformer script as a reference demonstration.

**Current milestone:** You have successfully trained your first PyTorch model and run a tiny Transformer text generator. The next structured milestone is building and testing your own tokenizer.

