import os
import torch

from .mcts import MCTSConfig, MCTS
from .model import MancalaTransformerConfig, MancalaTransformer


async def play(
        model_config: MancalaTransformerConfig,
        mcts_config: MCTSConfig,
        exp_dir: str,
        computer_first: bool = True
):
    model = MancalaTransformer.from_config(model_config)

    filename = None
    for file in os.listdir(exp_dir):
        if os.path.splitext(file)[1] == ".pt":
            filename = os.path.join(exp_dir, file)

    assert filename is not None, "No checkpoint found"
    model.load_state_dict(
        torch.load(filename, map_location="cpu")
    )

    mcts = MCTS(
        model,
        **mcts_config.__dict__
    )

    computer_move = computer_first
    while not mcts.root.state.terminal:
        mcts.root.state.display(int(computer_move))

        if computer_move:
            prior = await mcts.get_prior()

            action = prior.argmax().item()

            print(f"Computer move: {5 - action}")
        else:
            action = int(input("Enter your move: "))

        mcts.step(action)
        computer_move = (computer_move != mcts.root.state.flipped)

    mcts.root.state.display(int(computer_move))

    value = mcts.root.state.value
    if computer_move:
        value = -value

    if value == 0:
        print("It's a tie!")
    elif value == 1:
        print("Player wins!")
    else:
        print("Computer wins!")
