import torch

from torch import nn, LongTensor, FloatTensor
from math import sqrt
from typing import Tuple

from .. import NUM_PITS

from .util import Block, SinusoidalEmbedding, RMSNorm


class MancalaTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super(MancalaTransformer, self).__init__()
        self.emb = SinusoidalEmbedding(d_model)

        emb_scale = sqrt(1 / d_model)
        self.pits_pos_emb = nn.Parameter(
            torch.empty(1, 2, NUM_PITS, d_model).normal_(std=emb_scale)
        )
        self.score_pos_emb = nn.Parameter(
            torch.empty(1, 2, d_model).normal_(std=emb_scale)
        )
        self.state_value_emb = nn.Parameter(
            torch.empty(1, 1, d_model).normal_(std=emb_scale)
        )
        self.action_value_emb = nn.Parameter(
            torch.empty(1, 1, d_model).normal_(std=emb_scale)
        )

        self.layers = nn.Sequential(*[
            Block(d_model, n_heads) for _ in range(n_layers)
        ])

        self.state_value_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Tanh()
        )
        self.action_value_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, NUM_PITS)
        )

    def forward(self, score: LongTensor, pits: LongTensor) -> Tuple[FloatTensor, FloatTensor]:
        B = score.shape[0]
        x = torch.concatenate((
            self.state_value_emb.expand(B, -1, -1),
            self.action_value_emb.expand(B, -1, -1),
            self.score_pos_emb + self.emb(score),
            (self.pits_pos_emb + self.emb(pits)).flatten(1, 2)
        ), dim=1)

        x = self.layers(x)

        return self.state_value_head(x[:, 0]).squeeze(-1), self.action_value_head(x[:, 1])
