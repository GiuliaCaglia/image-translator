"""Training script."""

import click

from image_translator.trainers import trainer as t
from image_translator.utils.constants import Paths


@click.command()
@click.option("-lr", type=float, default=2e-2)
@click.option("-b", "--batch-size", type=int)
@click.option("-e", "--epochs", type=int)
@click.option("-d", "--device", type=str)
def train(epochs: int, batch_size: int, device: str, lr: float):
    trainer = t.Trainer()
    train_data, test_data = trainer.get_data(batch_size=batch_size)
    train_artifacts = trainer.fit(
        train_data, test_data, device=device, epochs=epochs, lr=lr
    )

    train_artifacts.dump_metrics(Paths.METRICS)
    train_artifacts.dump_model(Paths.MODEL)
