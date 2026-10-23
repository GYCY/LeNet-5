r"""Run with .venv\Scripts\python.exe check_gpu.py to verify CUDA training."""

import sys

import torch
import torchvision
from torch import nn


def main():
    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch build.")

    print("GPU:", torch.cuda.get_device_name(0))
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 10),
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 1, 32, 32, device="cuda")
    labels = torch.randint(0, 10, (8,), device="cuda")
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print("GPU forward/backward/optimizer step: PASS (loss={:.4f})".format(loss.item()))


if __name__ == "__main__":
    main()
