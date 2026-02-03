import time
from typing import Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.ao.quantization as tq
from tqdm import trange

def generate_data(n_samples: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic 2D classification dataset.

    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Features X and labels y.
    """

    X = torch.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] > 1).long()
    return X, y

X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(300)

class FP32Net(nn.Module):
    """
    Standard FP32 neural network with 2 hidden layers for classification.

    Architecture:
        Input -> Linear(2,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,2) -> Output
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FP32 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 2).
        """

        return self.net(x)

class QuantNet(nn.Module):
    """
    Neural network prepared for quantization using QuantStub and DeQuantStub.

    Architecture:
        QuantStub -> Linear(2,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,2) -> DeQuantStub
    """

    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantizable model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 2).
        """

        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 1e-3) -> None:
    """
    Train a PyTorch model using Adam optimizer and cross-entropy loss.

    Args:
        model (nn.Module): PyTorch model to train.
        X (torch.Tensor): Training features.
        y (torch.Tensor): Training labels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for Adam optimizer.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = trange(epochs, desc="Training")
    for epoch in pbar:
        model.train()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.write(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f}")

def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """
    Evaluate the accuracy of a model.

    Args:
        model (nn.Module): PyTorch model to evaluate.
        X (torch.Tensor): Features for evaluation.
        y (torch.Tensor): True labels.

    Returns:
        float: Accuracy in percentage.
    """

    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
        acc = (pred == y).float().mean().item() * 100
    return acc

def benchmark(model: nn.Module, X: torch.Tensor, runs: int = 100) -> float:
    """
    Measure the average inference time per forward pass.

    Args:
        model (nn.Module): PyTorch model to benchmark.
        X (torch.Tensor): Input tensor for inference.
        runs (int): Number of runs to average timing.

    Returns:
        float: Average inference time in seconds.
    """

    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(X)
        start = time.time()
        for _ in range(runs):
            _ = model(X)
    return (time.time() - start) / runs

"""
FP32 baseline
"""
fp32_model = FP32Net()
train_model(fp32_model, X_train, y_train)
fp32_acc = evaluate(fp32_model, X_test, y_test)
fp32_time = benchmark(fp32_model, X_test)
print(f"\nFP32 Accuracy: {fp32_acc:.2f}% | Time: {fp32_time*1000:.3f} ms")

"""
PTQ - Post Training Quantization
"""
ptq_model = QuantNet()
ptq_model.load_state_dict(fp32_model.state_dict())
ptq_model.eval()

ptq_model.qconfig = tq.get_default_qconfig("fbgemm")
tq.prepare(ptq_model, inplace=True)

with torch.no_grad():
    ptq_model(X_train)

tq.convert(ptq_model, inplace=True)

ptq_acc = evaluate(ptq_model, X_test, y_test)
ptq_time = benchmark(ptq_model, X_test)
print(f"PTQ Accuracy: {ptq_acc:.2f}% | Time: {ptq_time*1000:.3f} ms")

"""
QAT - Quantization Aware Training
"""
qat_model = QuantNet()
qat_model.load_state_dict(fp32_model.state_dict())
qat_model.train()

qat_model.qconfig = tq.get_default_qat_qconfig("fbgemm")
tq.prepare_qat(qat_model, inplace=True)
train_model(qat_model, X_train, y_train)
qat_model.eval()
tq.convert(qat_model, inplace=True)

qat_acc = evaluate(qat_model, X_test, y_test)
qat_time = benchmark(qat_model, X_test)
print(f"QAT Accuracy: {qat_acc:.2f}% | Time: {qat_time*1000:.3f} ms")


"""
Dynamic Quantization
"""
dyn_model = tq.quantize_dynamic(
    fp32_model,
    {nn.Linear},
    dtype=torch.qint8
)
dyn_acc = evaluate(dyn_model, X_test, y_test)
dyn_time = benchmark(dyn_model, X_test)
print(f"Dynamic Accuracy: {dyn_acc:.2f}% | Time: {dyn_time*1000:.3f} ms")


print("\n=== ✅ Summary ===")
print(f"{'Model':25} | {'Accuracy (%)':12} | {'Inference Time (ms)':20}")
print("-"*65)
print(f"{'FP32':25} | {fp32_acc:12.2f} | {fp32_time*1000:20.2f}")
print(f"{'PTQ':25} | {ptq_acc:12.2f} | {ptq_time*1000:20.2f}")
print(f"{'QAT':25} | {qat_acc:12.2f} | {qat_time*1000:20.2f}")
print(f"{'Dynamic Quantization':25} | {dyn_acc:12.2f} | {dyn_time*1000:20.2f}")
