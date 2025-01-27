import numpy as np
import torch

from torch.utils.data import Dataset


class GameHistoryDataset(Dataset):
    def __init__(self, files):
        histories = [
            np.load(file, allow_pickle=True).item() for file in files
        ]

        self.scores = np.concatenate([
            history["scores"] for history in histories
        ])
        self.pits = np.concatenate([
            history["pits"] for history in histories
        ])
        self.mcst_probs = np.concatenate([
            history["mcts_probs"] for history in histories
        ])
        self.results = np.concatenate([
            history["results"] for history in histories
        ])

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.scores[idx]).to(torch.long),
            torch.from_numpy(self.pits[idx]).to(torch.long),
            torch.from_numpy(self.mcst_probs[idx]).to(torch.float32),
            torch.tensor(self.results[idx]).to(torch.float32)
        )
