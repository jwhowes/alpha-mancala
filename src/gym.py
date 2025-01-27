from __future__ import annotations

import torch

from typing import Optional
from dataclasses import dataclass
from torch import LongTensor, BoolTensor

from . import NUM_PITS


@dataclass
class State:
    pits: LongTensor
    score: LongTensor
    flipped: bool = False

    def display(self, player: 0 | 1):
        if player == 0:
            print("\t".join([str(p.item()) for p in self.pits[1]]))
            print("\t".join([str(p.item()) for p in self.pits[0]]))
        else:
            print("\t".join([str(p.item()) for p in self.pits[0]][::-1]))
            print("\t".join([str(p.item()) for p in self.pits[1]][::-1]))
        print(f"{str(self.score[player].item())}\t{str(self.score[1 - player].item())}")

    def step(self, action: int) -> State:
        assert not self.terminal, "Cannot perform action on a terminal property"
        assert 0 <= action < NUM_PITS, "Action out of range"

        pits = self.pits.clone()
        score = self.score.clone()

        count = pits[0, action].clone()
        pits[0, action] = 0

        side = 0
        while count > 0:
            action += -2 * side + 1

            if 0 <= action < NUM_PITS:
                pits[side, action] += 1
                count -= 1
            elif action == 6 and side == 0:
                score[0] += 1
                count -= 1

            if not (-1 <= action <= 6):
                side = 1 - side

        if side == 0 and action < NUM_PITS and pits[0, action] == 1 and pits[1, action] > 0:
            score[0] += pits[0, action] + pits[1, action]
            pits[0, action] = 0
            pits[1, action] = 0

        flipped = False

        if pits[0].max() == 0:
            score[1] += pits[1].sum()
            pits[1] = 0
        elif side == 1 or action < NUM_PITS:
            flipped = True
            pits = torch.flip(pits[[1, 0]], dims=(1,))
            score = score[[1, 0]]

        return State(pits=pits, score=score, flipped=flipped)

    @property
    def terminal(self) -> bool:
        return self.pits.max() == 0

    @property
    def value(self) -> Optional[-1 | 0 | 1]:
        if self.score[0] == self.score[1]:
            return 0

        return -2 * self.score.argmax() + 1 if self.terminal else None

    @staticmethod
    def initial() -> State:
        return State(
            pits=torch.full((2, NUM_PITS), 4, dtype=torch.long),
            score=torch.full((2,), 0, dtype=torch.long)
        )

    def illegal_moves(self) -> BoolTensor:
        return self.pits[0] == 0


if __name__ == "__main__":
    state = State.initial()
    print(state.step(5))
