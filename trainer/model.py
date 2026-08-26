"""Policy/WDL/material network for the chess bot.

Input contract (must match engine/src/encoder.h and nn_client.h):
    planes : float [B, 113, 8, 8]   occupancy maps in {0,1} (112 age-block + ep)
    scalars: float [B, 9]           halfmove, fullmove, rep, stm, 4x castling, ep

Output contract (consumed by engine/src/nn_client.cpp):
    policy  : float [B, 4672]       logits; index = plane*64 + from_square
    wdl     : float [B, 3]          logits (win/draw/loss from stm perspective)
    material: float [B, 1]          predicted material diff (stm perspective)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLANES = 113
NUM_SCALARS = 9
IN_CHANNELS = NUM_PLANES + NUM_SCALARS   # scalars broadcast as extra channels
POLICY_ACTIONS = 73 * 64                 # 4672


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock(nn.Module):
    def __init__(self, ch: int, use_se: bool = True):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.se = SEBlock(ch) if use_se else None

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        if self.se is not None:
            y = self.se(y)
        return F.relu(x + y, inplace=True)


class ChessNet(nn.Module):
    def __init__(self, blocks: int = 10, channels: int = 160):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head -> 73 channels flattened channel-major (plane*64 + sq)
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 73, 1),
        )
        # WDL value head
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )
        # Auxiliary material head
        self.m_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor):
        s = scalars[:, :, None, None].expand(-1, -1, 8, 8)
        x = torch.cat([planes, s], dim=1)

        z = self.body(self.stem(x))
        policy = self.p_head(z).flatten(1)          # [B, 4672]
        wdl = self.v_head(z)                        # [B, 3]
        material = self.m_head(z)                   # [B, 1]
        return policy, wdl, material


def build_model(cfg: dict) -> ChessNet:
    return ChessNet(blocks=cfg.get("blocks", 10), channels=cfg.get("channels", 160))
