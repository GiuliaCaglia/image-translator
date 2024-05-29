"""Training script."""

import click

from image_translator.trainers import trainer as t
from image_translator.utils.constants import Paths


@click.command()
def train():
    trainer = t.Trainer()
    train_data, test_data = trainer.get_data()
    train_artifacts = trainer.fit(train_data, test_data)

    train_artifacts.dump_metrics(Paths.METRICS)
    train_artifacts.dump_model(Paths.MODEL)
    train_artifacts.dump_train_samples(Paths.TRAIN_SAMPLES)
    train_artifacts.dump_test_samples(Paths.TEST_SAMPLES)
    train_artifacts.dump_loss_plot(Paths.LOSS_PLOT)
