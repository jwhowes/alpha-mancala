from __future__ import annotations

import asyncio
import torch
import torch.nn.functional as F
import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, List

from .gym import State
from .model import MancalaTransformer
from . import NUM_PITS
from .util import Config


@dataclass
class MCTSConfig(Config):
    num_threads: int = 32
    sims_per_move: int = 50

    explore_coeff: float = 5.0
    virtual_loss: float = 3.0
    temperature: float = 1.0


class ModelQueue:
    def __init__(self, num_threads: int, device: torch.device):
        self.num_threads = num_threads
        self.device = device
        
        self.queue = asyncio.Queue()
        self.queue_full = asyncio.Event()
        self.completed = asyncio.Event()
        self.responses = {}

    async def add_to_queue(self, state: Optional[State], thread_id: int):
        await self.queue.put((state, thread_id))

        if self.queue.qsize() == self.num_threads:
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
                    ]).to(self.device),
                    pits=torch.stack([
                        state.pits for state in states
                    ]).to(self.device)
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

    async def search(self, thread_id: int, queue: ModelQueue, virtual_loss: float = 3.0, explore_coeff: float = 5.0) -> Optional[float]:
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

        if all([c.locked for c in self.children]):
            await queue.add_to_queue(None, thread_id)

            return None

        quality = (
                np.nan_to_num(self.total_value / self.num_visits) +
                explore_coeff * self.prior * np.sqrt(self.num_visits.sum()) / (1 + self.num_visits)
        )

        child_idx = quality.argmax()
        self.num_visits[child_idx] += virtual_loss
        self.total_value[child_idx] -= virtual_loss

        value = await self.children[child_idx].search(thread_id, queue, virtual_loss, explore_coeff)

        self.num_visits[child_idx] -= virtual_loss
        self.total_value[child_idx] += virtual_loss

        if value is not None:
            self.num_visits[child_idx] += 1

            if self.children[child_idx].state.flipped:
                value = -value

            self.total_value[child_idx] += value

        return value



class MCTS:
    def __init__(
            self,
            model: MancalaTransformer,
            num_threads: int,
            temperature: float,
            sims_per_move: float,
            explore_coeff: float,
            virtual_loss: float,
            device: torch.device
    ):
        self.device = device

        self.model = model
        self.num_threads = num_threads
        self.recip_temperature = 1.0 / temperature
        self.sims_per_move = sims_per_move
        self.explore_coeff = explore_coeff
        self.virtual_loss = virtual_loss

        self.submit_event = asyncio.Event()


        self.root = Node(
            state=State.initial()
        )

    async def simulation(self) -> None:
        queue = ModelQueue(num_threads=self.num_threads, device=self.device)

        workers = [
            self.root.search(i, queue, self.virtual_loss, self.explore_coeff) for i in range(self.num_threads)
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


async def self_play(model: MancalaTransformer, config: MCTSConfig, device: torch.device) -> GameHistory:
    mcts = MCTS(
        model,
        num_threads=config.num_threads,
        sims_per_move=config.sims_per_move,
        temperature=config.temperature,
        explore_coeff=config.explore_coeff,
        virtual_loss=config.virtual_loss,
        device=device
    )

    states: List[State] = []
    priors: List[NDArray[np.float32]] = []
    results: List[int] = []
    while not mcts.root.state.terminal:
        states.append(mcts.root.state)

        prior = await mcts.get_prior()
        priors.append(prior)

        action = np.random.multinomial(1, prior).argmax()
        mcts.step(action)

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
