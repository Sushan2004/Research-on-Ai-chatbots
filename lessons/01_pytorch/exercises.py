"""Stage 1: replace each TODO, then run one exercise at a time.

Usage: python lessons/01_pytorch/exercises.py tensors
Choices: tensors, autograd, train
"""
import argparse
import torch
from torch import nn


def tensors():
    # TODO: Create a float32 tensor containing 0..5, shaped (2, 3).
    x = torch.arange(6, dtype=torch.float32).reshape(2,3)
    # TODO: Create a float32 tensor of ones, shaped (3, 2).
    w = torch.ones((3,2),dtype=torch.float32)
    # TODO: Matrix-multiply x and w; then add a length-2 bias [10, 20].
    y = x @ w + torch.tensor([10.0, 20.0])

    assert x.shape == (2, 3) and x.dtype == torch.float32
    assert w.shape == (3, 2) and w.dtype == torch.float32
    torch.testing.assert_close(y, torch.tensor([[13., 23.], [22., 32.]]))
    print('Tensor checks passed:', y)
    # Experiment: try x * w. Explain the resulting error before changing shapes.


def autograd():
    # TODO: Create scalar x=3.0 with gradient tracking enabled.
    x = torch.tensor(3.0, requires_grad=True)
    # TODO: Compute y = x**2 + 2*x + 1. Predict dy/dx on paper first.
    y = x**2+2*x+1
    # TODO: Ask autograd to backpropagate from y.
    y.backward()
    assert x.grad is not None, 'No gradient: did you call backward()?'
    torch.testing.assert_close(y.detach(), torch.tensor(16.))
    torch.testing.assert_close(x.grad, torch.tensor(8.))
    print('Autograd checks passed. Gradient:', x.grad.item())
    # Experiment: recompute y and backpropagate again without clearing x.grad.
    # Predict the new gradient, then check it. Why recompute the graph?


class LineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
     


def train(device_name):
    # Make the random initialization repeatable.
    torch.manual_seed(42)

    # Choose where calculations happen: CPU or GPU.
    device = torch.device(device_name)

    # Stop if CUDA was requested but isn't available.
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with --device cpu.")

    # Create 128 input numbers between -1 and 1.
    # Shape (128, 1): 128 examples, each with one input feature.
    x = torch.linspace(-1, 1, 128).reshape(-1, 1).to(device)

    # Calculate the correct answer for each input.
    target = 3 * x + 2

    # Create the model and move its parameters to the same device as x.
    model = LineModel().to(device)

    # SGD will adjust the model's weight and bias.
    # The learning rate controls the size of each update.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Mean squared error measures the average squared prediction error.
    loss_fn = nn.MSELoss()

    # Store loss values so we can inspect training progress.
    history = []

    # Train using all 128 examples for each of 201 updates.
    for step in range(201):
        # Clear previous gradients because PyTorch accumulates them.
        optimizer.zero_grad()

        # Predict answers using the current weight and bias.
        prediction = model(x)

        # Compare predictions with the correct answers.
        loss = loss_fn(prediction, target)

        # TODO: Compute gradients by calling backward() on loss.
        loss.backward()
       

        # TODO: Update the weight and bias by calling step() on optimizer.
        optimizer.step()

        # Convert the scalar loss tensor into a Python number and save it.
        history.append(loss.item())

        # Display progress every 50 steps.
        if step % 50 == 0:
            print(f"step={step:3d} loss={loss.item():.6f}")


    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(x), target).item()
        prediction = model(torch.tensor([[2.0]], device=device)).item()
    assert final_loss < 1e-4, f'Loss is still high: {final_loss}'
    assert abs(prediction - 8.0) < 0.05, f'Prediction is {prediction}'
    assert all(p.device == x.device for p in model.parameters())
    print(f'Training checks passed on {device}. f(2)={prediction:.4f}')
    print('Learned parameters:', {k: v.detach().cpu().tolist()
                                 for k, v in model.named_parameters()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exercise', choices=['tensors', 'autograd', 'train'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.exercise == 'train':
        train(args.device)
    else:
        {'tensors': tensors, 'autograd': autograd}[args.exercise]()
