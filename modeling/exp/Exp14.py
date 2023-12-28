"""
Exp14.

* Add tids for X.

Author: JiaWei Jiang
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _WaveBlock(nn.Module):
    def __init__(self, n_layers: int, in_dim: int, h_dim: int) -> None:
        super().__init__()

        self.n_layers = n_layers

        dilation_rates = [2**i for i in range(n_layers)]
        self.in_conv = nn.Conv1d(in_dim, h_dim, 1, padding="same")
        self.filts = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            self.filts.append(nn.Conv1d(h_dim, h_dim, 3, padding="same", dilation=dilation_rates[layer]))
            self.gates.append(nn.Conv1d(h_dim, h_dim, 3, padding="same", dilation=dilation_rates[layer]))
            self.skip_convs.append(nn.Conv1d(h_dim, h_dim, 1, padding="same"))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, C_in, T)
        """
        x = self.in_conv(x)  # (B, H, T)
        x_resid = x

        for layer in range(self.n_layers):
            x_filt = self.filts[layer](x)
            x_gate = self.gates[layer](x)
            x = F.tanh(x_filt) * F.sigmoid(x_gate)
            x = self.skip_convs[layer](x)

            x_resid = x_resid + x

        return x_resid


class Exp(nn.Module):
    """Exp14."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        super().__init__()

        # Model blocks
        self.wave_block1 = _WaveBlock(8, 12, 64)
        self.wave_block2 = _WaveBlock(5, 64, 64)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.avg_pool = nn.AvgPool1d(24, stride=1)
        self.max_pool = nn.MaxPool1d(24, stride=1)

        self.last_gru = nn.GRU(159, 128, 1, batch_first=True, bidirectional=True)
        self.prod_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cons_head = nn.Sequential(
            nn.Linear(128, 64),
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
            x_tids: (B, 7, T)
            tids: (B, 7, 24)
            cli_attr: (B, 3)
            fwth: (B, C, 24)

        Returns:
            output: prediction
        """
        x, x_tids = inputs["x"], inputs["x_tids"]
        tids, cli_attr, fwth = inputs["tids"], inputs["cli_attr"], inputs["fwth"]
        batch_size, _, t_window = x.shape
        x_bypass = x

        cli_attr = cli_attr.unsqueeze(dim=-1).expand(-1, -1, t_window)
        x = torch.cat([x, x_tids, cli_attr], dim=1)  # (B, 12, T)
        x = self.wave_block1(x)  # (B, 64, T)
        x = self.bn1(x)
        x = self.wave_block2(x)
        x = self.bn2(x)

        x = torch.cat(
            [x[..., -24:], self.avg_pool(x[:, :32])[..., -24:], self.max_pool(x[:, 32:])[..., -24:]], axis=1
        )  # (B, 128, 24)

        x_bypass = x_bypass.reshape(batch_size, 2, 6, 24).reshape(batch_size, 12, 24)
        x = torch.cat([x, x_bypass, tids, fwth], dim=1)  # (B, 159, 24)
        x, _ = self.last_gru(x.transpose(1, 2))  # (B, 24, 2*128)
        output_p = self.prod_head(x[..., :128])
        output_c = self.cons_head(x[..., 128:])

        output = torch.concat([output_p, output_c], dim=-1).reshape(batch_size, -1)  # (B, 48)

        return output
