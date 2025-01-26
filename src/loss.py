import torch.nn.functional as F

from torch import FloatTensor


def alpha_zero_loss(state_value: FloatTensor, action_probs: FloatTensor, result: FloatTensor, mcts_probs: FloatTensor):
    return (
        F.binary_cross_entropy_with_logits(state_value, result) +
        -(mcts_probs * F.log_softmax(action_probs, dim=-1)).sum(-1).mean()
    )
