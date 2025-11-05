from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import trange


class CustomRNN(nn.Module):
    """
    Custom RNN module supporting unidirectional and bidirectional RNN.

    Attributes:
    - hidden_size: Number of hidden units in the RNN.
    - bidirectional: Whether to use a bidirectional RNN.
    """
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False) -> None:
        """
        Constructor
        :param input_size: Number of input features per timestep
        :param hidden_size: Number of hidden units
        :param bidirectional: Whether to use bidirectional RNN
        :return: None
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.Wxh_f = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh_f = nn.Parameter(torch.randn(hidden_size))

        if self.bidirectional:
            self.Wxh_b = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
            self.Whh_b = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
            self.bh_b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        :return:
        sequence_output: Hidden states for all timesteps (batch_size, seq_len, hidden_size * num_directions)
        last_hidden: Hidden state for last timestep (batch_size, hidden_size * num_directions)
        """
        batch_size, seq_len, _ = x.size()
        h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs_f = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h_f = torch.tanh(xt @ self.Wxh_f.T + h_f @ self.Whh_f + self.bh_f)
            outputs_f.append(h_f.unsqueeze(1))
        outputs_f = torch.cat(outputs_f, dim=1)

        if self.bidirectional:
            h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            outputs_b = []
            for t in reversed(range(seq_len)):
                xt = x[:, t, :]
                h_b = torch.tanh(xt @ self.Wxh_b.T + h_b @ self.Whh_b.T + self.bh_b)
                outputs_b.insert(0, h_b.unsqueeze(1))
            outputs_b = torch.cat(outputs_b, dim=1)
            sequence_output = torch.cat([outputs_f, outputs_b], dim=2)
            last_hidden = torch.cat([h_f, h_b], dim=1)
        else:
            sequence_output = outputs_f
            last_hidden = h_f

        return sequence_output, last_hidden

class RNNModel(nn.Module):
    """
    Simple RNN model combining CustomRNN and a linear output layer
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 10, bidirectional: bool =False) -> None:
        """
        Constructor
        :param input_size: Number of input features per timestep
        :param hidden_size: Number of hidden units in RNN
        :param bidirectional: Whether to use bidirectional RNN
        :return: None
        """
        super().__init__()
        self.rnn = CustomRNN(input_size, hidden_size, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RNN model.
        :param x: Input tensor (batch_size, seq_len, input_size)
        :return: Output tensor (batch_size, seq_len, 1)
        """
        out_seq, _ = self.rnn(x)
        out = self.linear(out_seq)
        return out

def generate_sine_sequences(seq_len: int =10, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic sine wave sequences for next-step prediction.
    :param seq_len: Length of each sequence
    :param n_samples: Number of sequences to generate
    :return: Tuple of X (inputs) and y (targets)
    """
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.rand() * 2 * np.pi
        seq = np.sin(np.linspace(start, start + seq_len * 0.1, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[1:].reshape(-1, 1))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 0.01) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in trange(epochs, desc=f"Training {model.__class__.__name__}"):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    return model

def train_torch_rnn(rnn: nn.RNN, linear: nn.Linear, X: torch.Tensor, y: torch.Tensor,
                    epochs=50, lr=0.01):
    """
    Trains a model using Adam optimizer and MSE loss.
    :param model: nn.Module instance
    :param X: Input data (batch_size, seq_len, input_size)
    :param y: Target data (batch_size, seq_len, 1)
    :param epochs: Number of epochs
    :param lr: Learning rate
    :return: Trained model
    """
    optimizer = optim.Adam(list(rnn.parameters()) + list(linear.parameters()), lr=lr)
    criterion = nn.MSELoss()
    for _ in trange(epochs, desc="Training torch.RNN"):
        optimizer.zero_grad()
        out, _ = rnn(X)
        out = linear(out)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    return rnn, linear

def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes Mean Squared Error on provided data.
    :param model: nn.Module instance
    :param X: Input tensor
    :param y: Target tensor
    :return: MSE loss value
    """
    with torch.no_grad():
        pred = model(X)
        return nn.MSELoss()(pred, y).item()

X_train, y_train = generate_sine_sequences()
X_test, y_test = generate_sine_sequences(200)

custom_rnn = train_model(RNNModel(hidden_size=10, bidirectional=False), X_train, y_train)
custom_birnn = train_model(RNNModel(hidden_size=10, bidirectional=True), X_train, y_train)

rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
linear = nn.Linear(10, 1)
rnn, linear = train_torch_rnn(rnn, linear, X_train, y_train)

bidir_rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True, bidirectional=True)
linear_bidir = nn.Linear(20, 1)
bidir_rnn, linear_bidir = train_torch_rnn(bidir_rnn, linear_bidir, X_train, y_train)

print("\n=== 📊 MSE Comparison ===")
print(f"Custom RNN       : {evaluate(custom_rnn, X_test, y_test):.6f}")
print(f"Custom BiRNN     : {evaluate(custom_birnn, X_test, y_test):.6f}")
print(f"torch.nn.RNN     : {evaluate(lambda x: linear(rnn(x)[0]), X_test, y_test):.6f}")
print(f"torch.nn.BiRNN   : {evaluate(lambda x: linear_bidir(bidir_rnn(x)[0]), X_test, y_test):.6f}")