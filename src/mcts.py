from __future__ import annotations

import asyncio
import torch
import torch.nn.functional as F
import numpy as np
import warnings

from numpy.typing import NDArray
from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional, List

from .gym import State
from .model import MancalaTransformer
from . import NUM_PITS


NUM_THREADS: int = 32
EXPLORE_CONST: float = 5.0
VIRTUAL_LOSS: float = 3.0


class ModelQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.queue_full = asyncio.Event()
        self.completed = asyncio.Event()
        self.responses = {}

    async def add_to_queue(self, state: Optional[State], thread_id: int):
        await self.queue.put((state, thread_id))

        if self.queue.qsize() == NUM_THREADS:
            self.queue_full.set()

    async def join(self):
        return self.queue.join()

    async def process_queue(self, model: MancalaTransformer):
        await self.queue_full.wait()

        thread_ids: List[int] = []
        states: List[State] = []
        while not self.queue.empty():
            state, thread_id = await self.queue.get()

            if state is not None:
                states.append(state)
                thread_ids.append(thread_id)

        if len(states) > 0:
            with torch.inference_mode():
                value, prior = model(
                    score=torch.stack([
                        state.score for state in states
                    ]),
                    pits=torch.stack([
                        state.pits for state in states
                    ])
                )

                for i, (state, thread_id) in enumerate(zip(states, thread_ids)):
                    prior[i, state.illegal_moves()] = float('-inf')
                    self.responses[thread_id] = (
                        torch.sigmoid(value[i]).cpu().numpy(), F.softmax(prior[i], dim=-1).cpu().numpy()
                    )

        self.completed.set()


class Node:
    def __init__(self, state: State):
        self.state = state

        self.locked = False

        self.children: Optional[List[Node]] = None
        self.prior: Optional[NDArray[np.float32]] = None

        self.num_visits: NDArray[np.int64] = np.zeros(NUM_PITS, dtype=np.int64)
        self.total_value: NDArray[np.float32] = np.zeros(NUM_PITS, dtype=np.float32)

    async def search(self, thread_id: int, queue: ModelQueue) -> Optional[float]:
        if self.state.terminal:
            await queue.add_to_queue(None, thread_id)

            return float(self.state.value)

        if self.children is None:
            self.locked = True
            await queue.add_to_queue(self.state, thread_id)
            await queue.completed.wait()

            value, self.prior = queue.responses[thread_id]

            self.children = [
                Node(state=self.state.step(action))
                for action in range(NUM_PITS)
            ]

            self.locked = False

            return value

        if all([c.locked for c in self.children]) or all([c.state.terminal for c in self.children]):
            await queue.add_to_queue(None, thread_id)

            return None

        quality = (
                np.nan_to_num(self.total_value / self.num_visits) +
                EXPLORE_CONST * self.prior * np.sqrt(self.num_visits.sum()) / (1 + self.num_visits)
        )

        child_idx = quality.argmax()
        self.num_visits[child_idx] += VIRTUAL_LOSS
        self.total_value[child_idx] -= VIRTUAL_LOSS

        value = await self.children[child_idx].search(thread_id, queue)

        if value is not None:
            self.num_visits[child_idx] += 1 - VIRTUAL_LOSS

            if self.children[child_idx].state.flipped:
                value = -value

            self.total_value[child_idx] += value + VIRTUAL_LOSS

        return value



class MCTS:
    def __init__(self, model: MancalaTransformer, temperature: float = 1.0, sims_per_move: int = 1000):
        self.model = model

        self.submit_event = asyncio.Event()

        self.recip_temperature = 1.0 / temperature
        self.sims_per_move = sims_per_move

        self.root = Node(
            state=State.initial()
        )

    async def simulation(self) -> None:
        queue = ModelQueue()

        workers = [
            self.root.search(i, queue) for i in range(NUM_THREADS)
        ]

        asyncio.create_task(queue.process_queue(self.model))

        await asyncio.gather(*workers)

    async def get_prior(self) -> NDArray[np.float32]:
        for _ in range(self.sims_per_move):
            await self.simulation()

        likelihood = self.root.num_visits ** self.recip_temperature

        return likelihood / likelihood.sum()

    def step(self, action: int):
        self.root = self.root.children[action]


@dataclass
class GameHistory:
    scores: NDArray[np.int64]
    pits: NDArray[np.int64]

    mcts_probs: NDArray[np.float32]

    results: NDArray[np.float32]


async def self_play(model: MancalaTransformer) -> GameHistory:
    mcts = MCTS(model, sims_per_move=50)

    states: List[State] = []
    priors: List[NDArray[np.float32]] = []
    results: List[int] = []

    pbar = tqdm()
    while not mcts.root.state.terminal:
        states.append(mcts.root.state)

        prior = await mcts.get_prior()
        priors.append(prior)

        action = np.random.multinomial(1, prior).argmax()
        mcts.step(action)

        pbar.update()

    result = mcts.root.state.value
    assert result is not None

    for state in states[::-1]:
        results = [result] + results
        if state.flipped:
            result = -result

    return GameHistory(
        scores=np.stack([
            state.score for state in states
        ]).astype(np.int64),
        pits=np.stack([
            state.pits for state in states
        ]).astype(np.int64),
        mcts_probs=np.stack(priors).astype(np.float32),
        results=np.stack(results).astype(np.float32)
    )


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    model = MancalaTransformer(256, 4, 4)

    history = asyncio.run(self_play(model))

    print(history.results)
    print(history.scores[-1])
