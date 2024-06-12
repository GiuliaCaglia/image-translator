"""Training script."""

import os

import click
import dotenv
import mlflow
import yaml

from image_translator.trainers import trainer as t
from image_translator.utils.constants import EXPERIMENT_NAME_KEY, MLFLOW_URI_KEY, Paths

dotenv.load_dotenv()

mlflow.set_tracking_uri(os.getenv(MLFLOW_URI_KEY))
mlflow.set_experiment(experiment_name=os.getenv(EXPERIMENT_NAME_KEY))


@click.command()
def train():
    with mlflow.start_run():
        trainer = t.CheckpointedTrainer()
        train_data, test_data = trainer.get_data()
        train_artifacts = trainer.fit(train_data, test_data)

        with Paths.ENCODER_CONFIG.open("r", encoding="utf-8") as f:
            mlflow.log_params({"encoder": yaml.safe_load(f)})
        with Paths.DECODER_CONFIG.open("r", encoding="utf-8") as f:
            mlflow.log_params({"decoder": yaml.safe_load(f)})
        with Paths.TRAIN_CONFIG.open("r", encoding="utf-8") as f:
            mlflow.log_params(yaml.safe_load(f))
        mlflow.log_metric("train_loss", train_artifacts.train_losses[-1])
        mlflow.log_metric("test_loss", train_artifacts.test_loss)

        train_artifacts.dump_metrics(Paths.METRICS)
        train_artifacts.dump_model(Paths.MODEL)
        train_artifacts.dump_train_samples(Paths.TRAIN_SAMPLES)
        train_artifacts.dump_test_samples(Paths.TEST_SAMPLES)
        train_artifacts.dump_loss_plot(Paths.LOSS_PLOT)
        mlflow.log_artifacts(Paths.RESULTS)
        mlflow.pytorch.log_model(train_artifacts.model, "autoencoder")
