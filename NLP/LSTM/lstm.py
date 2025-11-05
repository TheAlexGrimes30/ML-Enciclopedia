from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import trange


class CustomLSTM(nn.Module):
    """
    Custom implementation of an LSTM layer.
    Supports unidirectional and bidirectional modes.

    Methods:
        __init__: constructor, initializes weights
        lstm_cell: computes one LSTM step (one time step update)
        forward: processes full input sequence
    """

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        """
        Constructor for CustomLSTM.
        :param input_size: dimension of each input vector x_t
        :param hidden_size: size of the hidden state h_t
        :param bidirectional: whether to enable bidirectional processing
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.W_f = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.01)
        self.U_f = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(4 * hidden_size))

        if self.bidirectional:
            self.W_b = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.01)
            self.U_b = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)
            self.b_b = nn.Parameter(torch.zeros(4 * hidden_size))

    def lstm_cell(self, x: torch.Tensor, h: torch.Tensor,
                  c: torch.Tensor, W: torch.Tensor,
                  U: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one time-step computation of LSTM.
        :param x: current input vector, shape (batch_size, input_size)
        :param h: previous hidden state, shape (batch_size, hidden_size)
        :param c: previous cell state, shape (batch_size, hidden_size)
        :param W: input-to-hidden weights
        :param U: hidden-to-hidden weights
        :param b: bias vector
        :return: new hidden and cell states (h_t, c_t)
        """

        z = x @ W.T + h @ U.T + b
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the entire input sequence.
        :param x: input tensor of shape (batch_size, seq_len, input_size)
        :return: output sequence and final hidden state
        """

        batch_size, seq_len, _ = x.size()
        h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs_f = []

        for t in range(seq_len):
            xt = x[:, t, :]
            h_f, c_f = self.lstm_cell(xt, h_f, c_f, self.W_f, self.U_f, self.b_f)
            outputs_f.append(h_f.unsqueeze(1))

        outputs_f = torch.cat(outputs_f, dim=1)

        if self.bidirectional:
            h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            outputs_b = []

            for t in reversed(range(seq_len)):
                xt = x[:, t, :]
                h_b, c_b = self.lstm_cell(xt, h_b, c_b, self.W_b, self.U_b, self.b_b)
                outputs_b.insert(0, h_b.unsqueeze(1))
            outputs_b = torch.cat(outputs_b, dim=1)
            sequence_output = torch.cat([outputs_f, outputs_b], dim=2)
            last_hidden = torch.cat([h_f, h_b], dim=1)
        else:
            sequence_output = outputs_f
            last_hidden = h_f

        return sequence_output, last_hidden

class LSTMModel(nn.Module):
    """
    Neural network model combining CustomLSTM and a linear layer.
    Used for sequence-to-sequence or time-series prediction.

    Methods:
        __init__: constructor
        forward: runs full network on input
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 10, bidirectional: bool = False):
        """
        Constructor for LSTMModel.
        :param input_size: dimension of each input vector in the sequence
        :param hidden_size: number of hidden units in LSTM
        :param bidirectional: whether to enable bidirectional mode
        """

        super().__init__()
        self.lstm = CustomLSTM(input_size, hidden_size, bidirectional)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: input sequence tensor
        :return: predicted sequence
        """

        out_seq, _ = self.lstm(x)
        return self.linear(out_seq)

def generate_sine_sequences(seq_len: int = 10, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
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

def train_torch_lstm(lstm: nn.LSTM, linear: nn.Linear, X: torch.Tensor,
                     y: torch.Tensor, epochs=50, lr=0.01) -> Tuple[nn.LSTM, nn.Linear]:
    optimizer = optim.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=lr)
    criterion = nn.MSELoss()
    for _ in trange(epochs, desc="Training torch.LSTM"):
        optimizer.zero_grad()
        out, _ = lstm(X)
        out = linear(out)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    return lstm, linear

def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(X)
        return nn.MSELoss()(pred, y).item()

X_train, y_train = generate_sine_sequences()
X_test, y_test = generate_sine_sequences(200)

custom_lstm = train_model(LSTMModel(hidden_size=10, bidirectional=False), X_train, y_train)
custom_bilstm = train_model(LSTMModel(hidden_size=10, bidirectional=True), X_train, y_train)

lstm = nn.LSTM(input_size=1, hidden_size=10, batch_first=True)
linear = nn.Linear(10, 1)
lstm, linear = train_torch_lstm(lstm, linear, X_train, y_train)

bilstm = nn.LSTM(input_size=1, hidden_size=10, batch_first=True, bidirectional=True)
linear_bidir = nn.Linear(20, 1)
bilstm, linear_bidir = train_torch_lstm(bilstm, linear_bidir, X_train, y_train)

print("\n=== 📊 LSTM MSE Comparison ===")
print(f"Custom LSTM      : {evaluate(custom_lstm, X_test, y_test):.6f}")
print(f"Custom BiLSTM    : {evaluate(custom_bilstm, X_test, y_test):.6f}")
print(f"torch.nn.LSTM    : {evaluate(lambda x: linear(lstm(x)[0]), X_test, y_test):.6f}")
print(f"torch.nn.BiLSTM  : {evaluate(lambda x: linear_bidir(bilstm(x)[0]), X_test, y_test):.6f}")