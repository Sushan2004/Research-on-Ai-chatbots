# Stage 1: PyTorch fundamentals

Status: awaiting your attempts and test results. Do not advance to tokenization
until you confirm this stage is built and tested. Solutions are intentionally absent.

## Setup checkpoint (Windows PowerShell)

If Python is missing, install Python 3.12 from https://www.python.org/downloads/windows/
with the launcher and PATH option enabled, then open a fresh terminal.
From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The CPU wheel is intentional. CUDA reporting False is expected here. Use the
official installer selector at https://pytorch.org/get-started/locally/ later if
you have an NVIDIA GPU and want a compatible CUDA build. No GPU is required.
Using the virtual environment's executable directly avoids activation issues.

## Exercises

Edit `exercises.py`; fill only the section you are currently attempting.
Ellipses and NotImplementedError are deliberate placeholders, so the file will
not pass until you implement the TODOs. Start with `tensors` and share your attempt.

1. **Tensors:** multidimensional arrays with a shape, dtype, and device.
   Implement creation, reshape, matrix multiplication, and bias broadcasting.
   API hints: `torch.arange`, `torch.ones`, `torch.tensor`, `.reshape`, `@`.
   Before running: what shape should `(2, 3) @ (3, 2)` produce, and why can
   a length-2 bias be added to the result?
2. **Autograd:** operations on gradient-tracked tensors build a graph; backward
   computes derivatives and accumulates them in leaf tensors' `.grad` fields.
   Implement a scalar polynomial and differentiate it. API hints:
   `requires_grad=True`, `.backward()`, `.grad`.
   Before running: derive the derivative. Afterward: explain why gradients
   must be cleared between optimization steps.
3. **Module + training:** an `nn.Module` groups parameters and defines a forward
   calculation. An optimizer updates those parameters using gradients from a
   scalar loss. Fit the synthetic relationship `y = 3*x + 2` using `nn.Linear`.
   API hints: `.to(device)`, `torch.optim.SGD`, `.zero_grad()`, `.step()`.
   Think through the order: clear gradients, predict, measure error, backward,
   update. Explain why loss.backward() alone does not change the weights.
4. **CPU vs GPU:** model parameters and inputs must share a device. GPU parallelism
   helps large workloads, but overhead can make tiny workloads slower. This CPU
   exercise does not require CUDA. Inspect `x.device` and the parameter devices.
   Optional, only with a CUDA build and compatible GPU: repeat with `--device cuda`.

Run each independently from the repository root:

```powershell
.\.venv\Scripts\python.exe lessons/01_pytorch/exercises.py tensors
.\.venv\Scripts\python.exe lessons/01_pytorch/exercises.py autograd
.\.venv\Scripts\python.exe lessons/01_pytorch/exercises.py train
```

## Completion evidence

Share your code attempt, outputs (or traceback), and answers to the questions.
Training should reach MSE below 0.0001 and predict approximately 8 for x=2.
Explain what the learned weight and bias represent, and the difference between
`model.eval()` and `torch.no_grad()`. We will review your attempt before revealing
solutions, and wait for your explicit confirmation before Stage 2.

Reference: https://docs.pytorch.org/tutorials/beginner/basics/
