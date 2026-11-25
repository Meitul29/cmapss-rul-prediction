import torch
from torch import nn


class SequenceRULNet(nn.Module):
    """Recurrent sequence model (LSTM or GRU) with LayerNorm and SiLU head."""

    def __init__(
        self,
        num_features: int,
        cell_type: str = "gru",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        cell = cell_type.lower()
        if cell == "gru":
            rnn_cls = nn.GRU
        elif cell == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError(
                f"Unsupported cell_type '{cell_type}'. Choose 'lstm' or 'gru'."
            )

        self.rnn = rnn_cls(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        # Take the final time-step representation across the sequence window
        h_last = self.drop(self.norm(out[:, -1, :]))
        return self.head(h_last).squeeze(-1)


class RULSTM1(nn.Module):
    """Baseline 2-layer LSTM with ReLU head from the initial FD001 experiment."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.head(last_step).squeeze(-1)