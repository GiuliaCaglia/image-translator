"""Training module for image translator."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dill
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from image_translator.data import datasets
from image_translator.networks import networks
from image_translator.utils.constants import Paths
from image_translator.utils.utils import TrainingParams, get_logger


class TrainArtifact:
    def __init__(
        self,
        model: nn.Module,
        train_losses: List[float],
        train_samples: torch.Tensor,
        test_loss: Optional[float] = None,
        test_samples: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.train_losses = train_losses
        self.test_loss = test_loss
        self.train_samples = make_grid(train_samples)
        if test_samples is not None:
            self.test_samples = make_grid(test_samples)
        else:
            self.test_samples = None

    def get_metrics(self) -> Dict[str, Union[float, List[float], None]]:
        return {
            "train_losses": self.train_losses,
            "test_loss": self.test_loss,
        }

    def dump_metrics(self, path: Path):
        metrics = self.get_metrics()
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f)

    def dump_model(self, path: Path):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("wb") as f:
            dill.dump(self.model, f)

    def dump_train_samples(self, path: Path) -> Image:
        self._make_samples_grid(self.train_samples)
        plt.savefig(path)

    def dump_test_samples(self, path: Path) -> Image:
        self._make_samples_grid(self.test_samples)
        plt.savefig(path)

    def dump_loss_plot(self, path: Path):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.train_losses)
        plt.savefig(path)

    def _make_samples_grid(self, images: torch.Tensor) -> plt.Figure:
        grid = make_grid(images).detach().cpu()
        fig, ax = plt.subplots(1, 1)
        ax.imshow(F.to_pil_image(grid))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        return fig


class Trainer:

    LOG_EVERY = 10
    LOGGER = get_logger("Trainer")

    def __init__(self) -> None:
        self.encoder = networks.Coder.from_config(Paths.ENCODER_CONFIG)
        self.decoder = networks.Coder.from_config(Paths.DECODER_CONFIG)
        self.training_params = TrainingParams.load_yaml(Paths.TRAIN_CONFIG)

    def get_data(self, train_size: float = 0.9) -> Tuple[DataLoader, DataLoader]:
        train_images, test_images = datasets.TrainTestSplitPaths.get_split(
            train_size=train_size
        )
        max_index = 1 if self.training_params.smoke_test else -1
        train_dataset = datasets.ImageDataset(train_images[:max_index])
        test_dataset = datasets.ImageDataset(test_images[:max_index])
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_params.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.training_params.batch_size, shuffle=True
        )

        return train_loader, test_loader

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> TrainArtifact:
        autoencoder = networks.AutoEncoder(
            encoder=self.encoder,
            decoder=self.decoder,
            device=self.training_params.device,
        )
        train_losses = []
        optimizer = self.training_params.optimizer(
            lr=self.training_params.learning_rate, params=autoencoder.parameters()
        )
        epochs = self.training_params.epochs
        for epoch in tqdm(range(1, epochs + 1), total=epochs):
            epoch_loss = 0.0
            for original in train_loader:
                optimizer.zero_grad()
                reconstructed = autoencoder(original)
                batch_loss = self.training_params.loss_function(
                    reconstructed, original.to(self.training_params.device)
                )
                batch_loss.backward()
                optimizer.step()
                epoch_loss += float(batch_loss) / len(original)

            if epoch % self.LOG_EVERY == 0:
                self.LOGGER.info("Epoch %s/%s; loss: %s", epoch, epochs, epoch_loss)
            train_losses.append(epoch_loss)

        if test_loader is not None:
            with torch.no_grad():
                self.LOGGER.info("Getting test_loss...")
                test_loss = 0.0
                for batch in test_loader:
                    out = autoencoder(batch)
                    test_loss += float(
                        self.training_params.loss_function(
                            out, batch.to(self.training_params.device)
                        )
                    ) / len(batch)

                self.LOGGER.info("Test Loss: %s", test_loss)
        else:
            test_loss = None

        for i in train_loader:
            train_samples = autoencoder(i.clone())[:25]
            break
        for i in test_loader:
            test_samples = autoencoder(i.clone())[:25]
            break

        artifact = TrainArtifact(
            model=autoencoder,
            train_losses=train_losses,
            test_loss=test_loss,
            train_samples=train_samples,
            test_samples=test_samples,
        )

        return artifact
