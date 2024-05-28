"""Training module for image translator."""

import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple

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
from image_translator.utils.utils import get_logger


class TrainArtifact:
    def __init__(
        self,
        model: nn.Module,
        train_losses: List[float],
        train_samples: torch.Tensor,
        baseline_loss: Optional[float] = None,
        test_loss: Optional[float] = None,
        test_samples: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.train_losses = train_losses
        self.test_loss = test_loss
        self.baseline_loss = baseline_loss
        self.train_samples = make_grid(train_samples)
        if test_samples is not None:
            self.test_samples = make_grid(test_samples)
        else:
            self.test_samples = None

    def dump_metrics(self, path: Path):
        metrics = {
            "baseline_loss": self.baseline_loss,
            "train_losses": self.train_losses,
            "test_loss": self.test_loss,
        }
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

    CRITERION = nn.MSELoss()
    LOG_EVERY = 1
    LOGGER = get_logger("Trainer")

    def __init__(self) -> None:
        self.encoder_blocks = [
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.Flatten(),
            nn.Linear(in_features=128 * 16 * 16, out_features=128),
        ]
        self.decoder_blocks = [
            nn.Linear(in_features=128, out_features=128 * 16 * 16),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ]

    def get_data(
        self, train_size: float = 0.9, batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        train_images, test_images = datasets.TrainTestSplitPaths.get_split(
            train_size=train_size
        )
        train_dataset = datasets.ImageDataset(train_images[:1])
        test_dataset = datasets.ImageDataset(test_images[:1])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        lr: float = 2e-2,
        epochs: int = 10,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> TrainArtifact:
        encoder = networks.Coder(modules=self.encoder_blocks)
        decoder = networks.Coder(modules=self.decoder_blocks)
        autoencoder = networks.AutoEncoder(
            encoder=encoder, decoder=decoder, device=device
        )
        if test_loader is not None:
            self.LOGGER.info("Getting baseline_loss...")
            with torch.no_grad():
                baseline_loss = 0.0
                for batch in test_loader:
                    out = autoencoder(batch)
                    baseline_loss += float(self.CRITERION(out, batch.to(device))) / len(
                        batch
                    )

            self.LOGGER.info("Baseline_loss: %s", baseline_loss)
        else:
            baseline_loss = None

        train_losses = []
        optimizer = Adam(lr=lr, params=autoencoder.parameters())
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for original in tqdm(train_loader):
                optimizer.zero_grad()
                reconstructed = autoencoder(original)
                batch_loss = self.CRITERION(reconstructed, original.to(device))
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
                    test_loss += float(self.CRITERION(out, batch.to(device))) / len(
                        batch
                    )

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
            baseline_loss=baseline_loss,
            test_loss=test_loss,
            train_samples=train_samples,
            test_samples=test_samples,
        )

        return artifact
