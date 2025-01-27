import click
import os
import asyncio
import warnings

from src.mcts import MCTSConfig
from src.model import MancalaTransformerConfig
from src.train import TrainConfig, Trainer


@click.group(chain=True)
@click.pass_context
def cli(ctx):
    warnings.simplefilter("ignore")
    ctx.ensure_object(dict)


@cli.command()
@click.argument("exp-dir", type=click.Path(exists=True))
@click.option("--resume", is_flag=True, default=False, show_default=True)
def train(exp_dir: str, resume: bool):
    trainer = Trainer(
        MancalaTransformerConfig.from_yaml(os.path.join(exp_dir, "model.yaml")),
        MCTSConfig.from_yaml(os.path.join(exp_dir, "mcts.yaml")),
        TrainConfig.from_yaml(os.path.join(exp_dir, "train.yaml")),
        exp_dir,
        resume=resume
    )

    asyncio.run(trainer.train())


if __name__ == "__main__":
    cli()
