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
