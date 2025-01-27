import click
import os
import asyncio
import warnings

from src.mcts import MCTSConfig
from src.model import MancalaTransformerConfig
from src.train import TrainConfig, Trainer
from src.play import play as play_base


@click.group(chain=True)
@click.argument("exp-dir", type=click.Path(exists=True))
@click.pass_context
def cli(ctx, exp_dir: str):
    warnings.simplefilter("ignore")
    ctx.ensure_object(dict)

    ctx.obj["model_config"] = MancalaTransformerConfig.from_yaml(os.path.join(exp_dir, "model.yaml"))
    ctx.obj["mcts_config"] = MCTSConfig.from_yaml(os.path.join(exp_dir, "mcts.yaml"))
    ctx.obj["train_config"] = TrainConfig.from_yaml(os.path.join(exp_dir, "train.yaml"))

    ctx.obj["exp_dir"] = exp_dir


@cli.command()
@click.option("--resume", is_flag=True, default=False, show_default=True)
@click.pass_context
def train(ctx, resume: bool):
    trainer = Trainer(
        **ctx.obj,
        resume=resume
    )

    asyncio.run(trainer.train())


@cli.command()
@click.option("--computer-first", is_flag=True, default=True, show_default=True)
@click.pass_context
def play(ctx, computer_first: bool):
    asyncio.run(play_base(
        ctx.obj["model_config"],
        ctx.obj["mcts_config"],
        ctx.obj["exp_dir"],
        computer_first=computer_first
    ))


if __name__ == "__main__":
    cli()
