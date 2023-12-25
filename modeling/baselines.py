"""
Basline model architecture.
Author: JiaWei Jiang
"""
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class BaseTSModel(nn.Module):
    """Baseline time series model architecture."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        super().__init__()

        # Model blocks
        self.rnn = nn.GRU(input_size=2, hidden_size=64, num_layers=3, batch_first=True)
        self.lin = nn.Sequential(
            nn.Linear(67, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.prod_head = nn.Sequential(nn.Linear(71, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cons_head = nn.Sequential(nn.Linear(71, 32), nn.ReLU(), nn.Linear(32, 1))
        # self.prod_head[2].bias = nn.Parameter(torch.tensor([0.327143]))
        # self.cons_head[2].bias = nn.Parameter(torch.tensor([4.699539]))

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Args:
            inputs: model inputs

        Shape:
            x: (B, 2, T)
            tids: (B, 7, 24)
            inputs: (B, 3)

        Returns:
            output: prediction
        """
        x, tids, cli_attr = inputs["x"], inputs["tids"], inputs["cli_attr"]
        batch_size = x.shape[0]
        # x_lin = x.unsqueeze(dim=2).reshape(batch_size, 2, 6, 24).transpose(2, 3)  # (B, 2, 24, 6)

        x = x.transpose(1, 2)  # (B, T, 2)
        hs, _ = self.rnn(x)  # (B, T, 64), (3, B, 64)
        x_t = hs[:, -24:, :]

        x = torch.cat([x_t, cli_attr.unsqueeze(1).expand(-1, 24, -1)], dim=-1)  # (B, 24, 67)
        x = self.lin(x)  # (B, 24, 64)

        x_p = torch.cat([x, tids.transpose(1, 2)], dim=-1)
        x_c = torch.cat([x, tids.transpose(1, 2)], dim=-1)

        # output_p = self.prod_head(x_p) + self.prod_lin(x_lin[:, 0, ...])  # (B, 24, 1)
        output_p = self.prod_head(x_p)  # (B, 24, 1)
        # output_c = self.cons_head(x_c) + self.cons_lin(x_lin[:, 1, ...])  # (B, 24, 1)
        output_c = self.cons_head(x_c)  # (B, 24, 1)

        output = torch.concat([output_p, output_c], dim=-1).reshape(batch_size, -1)  # (B, 48)

        return output
