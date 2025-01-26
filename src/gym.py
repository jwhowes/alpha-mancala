from __future__ import annotations

import torch

from dataclasses import dataclass
from torch import LongTensor

from . import NUM_PITS


@dataclass
class State:
    pits: LongTensor
    score: LongTensor
    flipped: bool = False

    def step(self, action: int) -> State:
        assert not self.terminal, "Cannot perform action on a terminal property"
        assert 0 <= action < NUM_PITS, "Action out of range"

        pits = self.pits.clone()
        score = self.score.clone()

        count = pits[0, action].clone()
        pits[0, action] = 0

        side = 0
        while count > 0:
            action += 1

            if action < NUM_PITS:
                pits[side, action] += 1
                count -= 1
            else:
                if side == 0:
                    score[0] += 1
                    count -= 1

                side = 1 - side
                action = 0

        if side == 0 and action < NUM_PITS and pits[0, action] == 1:
            score[0] += pits[0, action] + pits[1, action]
            pits[0, action] = 0
            pits[1, action] = 0

        flipped = False

        if pits[0].max() == 0:
            score[1] += pits[1].sum()
            pits[1] = 0
        elif side == 1 or action < NUM_PITS:
            flipped = True
            pits = pits[[1, 0]]
            score = score[[1, 0]]

        return State(pits=pits, score=score, flipped=flipped)

    @property
    def terminal(self) -> bool:
        return self.pits.max(-1) == 0

    @property
    def winner(self) -> 0 | 1 | None:
        return self.score.argmax() if self.terminal else None

    @staticmethod
    def initial() -> State:
        return State(
            pits=torch.full((2, NUM_PITS), 4, dtype=torch.long),
            score=torch.full((2,), 0, dtype=torch.long)
        )
