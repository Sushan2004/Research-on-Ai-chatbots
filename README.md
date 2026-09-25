# Research on AI Chatbots

A hands-on research project exploring how language-model chatbots work by building their components in Python and PyTorch. The project starts with tensors and training loops, then progresses toward tokenization, Transformers, retrieval, and deployment.

## Current status

- Successfully trained a linear model on CPU to learn `y = 3*x + 2`.
- Implemented tensor and autograd exercises; their final test confirmation is pending.
- Ran a minimal character-level Transformer and generated text in a terminal.
- The Transformer is an exploratory demonstration. The staged implementation and understanding checks remain in progress.

The complete learning plan and progress report are in [Progess.md](Progess.md).

## Findings so far

### 1. A training loop can learn a simple relationship

The linear model began with a loss of `3.060877`. Its final learned parameters were:

| Measurement | Recorded result | Target |
|---|---|---|
| Weight | 2.9999983310699463 | 3 |
| Bias | 1.999999761581421 | 2 |
| Prediction for x = 2 | 8.0000 | 8 |

The training assertions passed. Loss printed as `0.000000` is rounded, not evidence of an exactly zero error. This experiment verifies the basic training pipeline; it does not establish language-model capability.

### 2. Text generation does not guarantee useful conversation

The tiny Transformer generated fragments resembling its training corpus, including phrases about its name and learning about Transformers. It also produced malformed words and continued prompts instead of reliably answering them.

These observations are consistent with memorization of a very small corpus. There is no held-out evaluation yet, so generalization has not been demonstrated.

### 3. Tokenizer coverage affects whether input is accepted

A prompt containing `?` caused `KeyError: '?'` because the character was absent from the training vocabulary. The current encoder looks up every character directly in a dictionary. An unsupported-character guard was proposed but is not present in the current `minimal_chatbot.py`.

This motivates testing punctuation, uppercase text, Unicode, and vocabulary coverage when building the tokenizer.

### 4. Training and inference need separate handling

Code review identified that the demonstration retrains on each launch, has no checkpoint-loading path, and does not switch to evaluation mode before chatting. It also uses independent prompts without conversation history.

Planned improvements include saving checkpoints, calling `model.eval()` for generation, and managing conversation context. These are pending changes, not completed features.

### 5. The Python environment matters

Using a different Python executable produced `ModuleNotFoundError: No module named 'torch'`. Running through the project's virtual environment resolved that problem. Installation and execution commands below use the same interpreter.

## Project files

| File | Purpose |
|---|---|
| [Progess.md](Progess.md) | Detailed roadmap, milestones, and progress report |
| [exercises.py](lessons/01_pytorch/exercises.py) | Tensor, autograd, and linear-model training exercises |
| [minimal_chatbot.py](minimal_chatbot.py) | Tiny character-level Transformer training and terminal generation demo |

## Setup on Windows

From the repository root in PowerShell, with Python installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy
```

The recorded experiments used Python 3.11 and PyTorch `2.14.0+cpu`. Dependencies are not yet pinned; the commands above install the available compatible versions. A GPU is not required for the initial exercises.

If the virtual environment is already configured, skip setup and use the run commands below.

## Run the exercises

```powershell
.\.venv\Scripts\python.exe lessons/01_pytorch/exercises.py tensors
.\.venv\Scripts\python.exe lessons/01_pytorch/exercises.py autograd
.\.venv\Scripts\python.exe lessons/01_pytorch/exercises.py train
```

Each command selects one exercise. Save any completed TODOs before running it.

## Run the Transformer demonstration

```powershell
.\.venv\Scripts\python.exe minimal_chatbot.py
```

The script trains for 3,000 steps before displaying `you:`. Enter `hello` to try generation, and press Ctrl+C to exit. Use characters present in the training corpus; unsupported characters currently terminate the program. Each restart trains a new model.

## Next research steps

1. Confirm the remaining PyTorch exercise results and understanding checks.
2. Implement a byte-level BPE tokenizer and test encode/decode round trips.
3. Build and test embeddings, positional information, and causal attention.
4. Train a decoder with held-out evaluation and saved checkpoints.
5. Add conversation handling, retrieval, an API, and Docker deployment.

For each experiment, record the question, configuration, observed results, limitations, and next action in the progress report. Advance through the learning stages after building and testing each one.
