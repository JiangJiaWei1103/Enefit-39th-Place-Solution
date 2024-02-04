"""
Exp1.

* Consider forecast weather.

Author: JiaWei Jiang
"""
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class Exp(nn.Module):
    """Exp1."""

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
        self.prod_head = nn.Sequential(
            nn.Linear(83, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cons_head = nn.Sequential(
            nn.Linear(83, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.prod_head[4].bias = nn.Parameter(torch.tensor([0.000388]))
        self.cons_head[4].bias = nn.Parameter(torch.tensor([0.161311]))

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Args:
            inputs: model inputs

        Shape:
            x: (B, 2, T)
            tids: (B, 7, 24)
            cli_attr: (B, 3)
            fwth: (B, C, 24)

        Returns:
            output: prediction
        """
        x, tids, cli_attr, fwth = inputs["x"], inputs["tids"], inputs["cli_attr"], inputs["fwth"]
        batch_size = x.shape[0]

        x = x.transpose(1, 2)  # (B, T, 2)
        hs, _ = self.rnn(x)  # (B, T, 64), (3, B, 64)
        x_t = hs[:, -24:, :]

        x = torch.cat([x_t, cli_attr.unsqueeze(1).expand(-1, 24, -1)], dim=-1)  # (B, 24, 67)
        x = self.lin(x)  # (B, 24, 64)

        x_p = torch.cat([x, tids.transpose(1, 2), fwth.transpose(1, 2)], dim=-1)  # (B, 24, 83)
        x_c = torch.cat([x, tids.transpose(1, 2), fwth.transpose(1, 2)], dim=-1)

        output_p = self.prod_head(x_p)  # (B, 24, 1)
        output_c = self.cons_head(x_c)  # (B, 24, 1)

        output = torch.concat([output_p, output_c], dim=-1).reshape(batch_size, -1)  # (B, 48)

        return output
