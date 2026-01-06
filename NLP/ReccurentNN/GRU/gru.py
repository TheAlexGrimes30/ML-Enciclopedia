from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import trange


class CustomGRU(nn.Module):
    """
    Custom implementation of a GRU layer.
    Supports unidirectional and bidirectional modes.

    Methods:
        __init__: initializes GRU parameters
        gru_cell: computes one GRU time-step update
        forward: processes input sequence and returns all hidden states + last hidden state
    """

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        """
        Constructor for CustomGRU.
        :param input_size: dimension of input vectors x_t
        :param hidden_size: size of hidden state h_t
        :param bidirectional: whether to use forward and backward directions
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.W_f = nn.Parameter(torch.randn(3 * hidden_size, input_size) * 0.01)
        self.U_f = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(3 * hidden_size))

        if self.bidirectional:
            self.W_b = nn.Parameter(torch.randn(3 * hidden_size, input_size) * 0.01)
            self.U_b = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) * 0.01)
            self.b_b = nn.Parameter(torch.zeros(3 * hidden_size))

    def gru_cell(self, x: torch.Tensor, h: torch.Tensor,
                 W: torch.Tensor, U: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Performs one GRU step.
        :param x: current input vector, shape (batch_size, input_size)
        :param h: previous hidden state, shape (batch_size, hidden_size)
        :param W: input-to-hidden weights
        :param U: hidden-to-hidden weights
        :param b: bias vector
        :return: new hidden state h_t
        """

        z, r, g = torch.chunk(x @ W.T + h @ U.T + b, 3, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        g = torch.tanh((x @ W[2 * self.hidden_size:].T) +
                       (r * h) @ U[2 * self.hidden_size:].T +
                       b[2 * self.hidden_size:])
        h_new = (1 - z) * h + z * g
        return h_new

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes an entire sequence through the GRU layer.
        :param x: input tensor of shape (batch_size, seq_len, input_size)
        :return:
            sequence_output: hidden states for all time steps, shape (batch, seq_len, hidden*directions)
            last_hidden: final hidden state(s), shape (batch, hidden*directions)
        """

        batch_size, seq_len, _ = x.size()
        h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs_f = []
        for t in range(seq_len):
            h_f = self.gru_cell(x[:, t, :], h_f, self.W_f, self.U_f, self.b_f)
            outputs_f.append(h_f.unsqueeze(1))
        outputs_f = torch.cat(outputs_f, dim=1)

        if self.bidirectional:
            h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            outputs_b = []
            for t in reversed(range(seq_len)):
                h_b = self.gru_cell(x[:, t, :], h_b, self.W_b, self.U_b, self.b_b)
                outputs_b.insert(0, h_b.unsqueeze(1))
            outputs_b = torch.cat(outputs_b, dim=1)

            sequence_output = torch.cat([outputs_f, outputs_b], dim=2)
            last_hidden = torch.cat([h_f, h_b], dim=1)
        else:
            sequence_output = outputs_f
            last_hidden = h_f

        return sequence_output, last_hidden

class GRUModel(nn.Module):
    """
    Wrapper model that applies the custom GRU layer followed by a linear output layer.
    Used for sequence-to-sequence regression tasks.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 10, bidirectional: bool = False):
        """
        :param input_size: dimension of input sequence elements
        :param hidden_size: GRU hidden dimension
        :param bidirectional: whether to use bidirectional GRU
        """

        super().__init__()
        self.gru = CustomGRU(input_size, hidden_size, bidirectional)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        """
        Forward pass through GRU and linear layer.
        :param x: input sequence (batch, seq_len, input_size)
        :return: predicted sequence (batch, seq_len, 1)
        """

        out_seq, _ = self.gru(x)
        return self.linear(out_seq)

def generate_sine_sequences(seq_len: int = 10, n_samples: int = 1000):
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.rand() * 2 * np.pi
        seq = np.sin(np.linspace(start, start + seq_len * 0.1, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[1:].reshape(-1, 1))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_model(model, X, y, epochs=50, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in trange(epochs, desc=f"Training {model.__class__.__name__}"):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def train_torch_gru(gru, linear, X, y, epochs=50, lr=0.01):
    optimizer = optim.Adam(list(gru.parameters()) + list(linear.parameters()), lr=lr)
    criterion = nn.MSELoss()
    for _ in trange(epochs, desc="Training torch.GRU"):
        optimizer.zero_grad()
        out, _ = gru(X)
        out = linear(out)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    return gru, linear


def evaluate(model, X, y):
    with torch.no_grad():
        return nn.MSELoss()(model(X), y).item()

X_train, y_train = generate_sine_sequences()
X_test, y_test = generate_sine_sequences(200)

custom_gru = train_model(GRUModel(hidden_size=10, bidirectional=False), X_train, y_train)
custom_bigru = train_model(GRUModel(hidden_size=10, bidirectional=True), X_train, y_train)

gru = nn.GRU(input_size=1, hidden_size=10, batch_first=True)
linear = nn.Linear(10, 1)
gru, linear = train_torch_gru(gru, linear, X_train, y_train)

bigru = nn.GRU(input_size=1, hidden_size=10, batch_first=True, bidirectional=True)
linear_bidir = nn.Linear(20, 1)
bigru, linear_bidir = train_torch_gru(bigru, linear_bidir, X_train, y_train)

print("\n=== 📊 GRU MSE Comparison ===")
print(f"Custom GRU      : {evaluate(custom_gru, X_test, y_test):.6f}")
print(f"Custom BiGRU    : {evaluate(custom_bigru, X_test, y_test):.6f}")
print(f"torch.nn.GRU    : {evaluate(lambda x: linear(gru(x)[0]), X_test, y_test):.6f}")
print(f"torch.nn.BiGRU  : {evaluate(lambda x: linear_bidir(bigru(x)[0]), X_test, y_test):.6f}")
