import asyncio
import os
import numpy as np
import torch

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from .data import GameHistoryDataset
from .loss import alpha_zero_loss
from .mcts import MCTSConfig, self_play
from .model import MancalaTransformer, MancalaTransformerConfig
from .util import Config, timestamp


@dataclass
class TrainConfig(Config):
    batch_size: int = 32

    queue_size: int = 16
    queue_min: int = 4

    data_dir: str = "data"

    lr: float = 5e-5
    weight_decay: float = 0.01
    clip_grad: Optional[float] = 3.0


class Trainer:
    def __init__(
            self,
            model_config: MancalaTransformerConfig,
            mcts_config: MCTSConfig,
            train_config: TrainConfig,
            exp_dir: str,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            resume: bool = False
    ):
        self.model_config = model_config
        self.mcts_config = mcts_config
        self.train_config = train_config
        self.exp_dir = exp_dir
        self.device = device

        self.data_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()

        self.queue = []

        if resume:
            self.queue = sorted([
                os.path.join(self.train_config.data_dir, file) for file in os.listdir(self.train_config.data_dir)
            ])[::-1]

            self.filename = None
            for file in os.listdir(exp_dir):
                if os.path.splitext(file)[1] == ".pt":
                    self.filename = os.path.join(exp_dir, file)
                    break

            assert self.filename is not None, "No model checkpoint found."
        else:
            for file in os.listdir(self.train_config.data_dir):
                os.remove(os.path.join(self.train_config.data_dir, file))

            for file in os.listdir(exp_dir):
                if os.path.splitext(file)[1] == ".pt":
                    os.remove(os.path.join(exp_dir, file))

            self.filename = os.path.join(self.exp_dir, f"{timestamp()}.pt")
            torch.save(
                MancalaTransformer.from_config(model_config).state_dict(),
                self.filename
            )
        self.data_fname = self.filename

    async def save_model(self, model: MancalaTransformer) -> str:
        async with self.model_lock:
            os.remove(self.filename)

            self.filename = os.path.join(self.exp_dir, f"{timestamp()}.pt")
            torch.save(
                model.state_dict(),
                self.filename
            )

    async def load_model(self, model: MancalaTransformer):
        async with self.model_lock:
            model.load_state_dict(torch.load(self.filename, map_location=self.device))

    async def train(self):
        await asyncio.gather(
            self.generate_data(),
            self.train_model()
        )

    async def train_model(self):
        model = MancalaTransformer.from_config(self.model_config).to(self.device)
        model.train()
        await self.load_model(model)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay
        )

        while True:
            if len(self.queue) < self.train_config.queue_min:
                await asyncio.sleep(1)
                continue

            async with self.data_lock:
                dataset = GameHistoryDataset(self.queue)

            dataloader = DataLoader(
                dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True
            )

            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader)
            )
            total_loss = 0
            for i, (score, pits, mcts_prob, result) in pbar:
                score = score.to(self.device)
                pits = pits.to(self.device)
                mcts_prob = mcts_prob.to(self.device)
                result = result.to(self.device)

                value, prob = model(score, pits)

                loss = alpha_zero_loss(value, prob, result, mcts_prob)
                loss.backward()

                total_loss += loss.item()

                pbar.set_description(f"Average Loss: {total_loss / (i + 1):.4f}")

                if not self.train_config.clip_grad is None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config.clip_grad)

                opt.step()

            await self.save_model(model)
            await asyncio.sleep(5)


    async def generate_data(self):
        model = MancalaTransformer.from_config(self.model_config).to(self.device)
        model.eval()
        await self.load_model(model)

        while True:
            if self.filename != self.data_fname:
                await self.load_model(model)
                self.data_fname = self.filename

            history = await self_play(model, self.mcts_config, self.device)

            async with self.data_lock:
                if len(self.queue) >= self.train_config.queue_size:
                    os.remove(self.queue.pop())

                fname = os.path.join(self.train_config.data_dir, f"{timestamp()}.npy")
                np.save(fname, history.__dict__)

                self.queue = [fname] + self.queue
