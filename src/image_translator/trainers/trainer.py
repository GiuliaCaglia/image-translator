"""Training module for image translator."""

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from image_translator.data import datasets
from image_translator.networks import networks


class TrainArtifact:
    def __init__(
        self,
        model: nn.Module,
        train_losses: List[float],
        baseline_loss: Optional[float] = None,
        test_loss: Optional[float] = None,
    ) -> None:
        self.model = model
        self.train_losses = train_losses
        self.test_loss = test_loss
        self.baseline_loss = baseline_loss


class Trainer:

    TEST_CRITERION = nn.MSELoss()

    def get_data(
        self, train_size: float = 0.9, batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        train_images, test_images = datasets.TrainTestSplitPaths.get_split(
            train_size=train_size
        )
        train_dataset = datasets.ImageDataset(train_images)
        test_dataset = datasets.ImageDataset(test_images)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> TrainArtifact:
        encoder = networks.Encoder()
        decoder = networks.Decoder()
        autoencoder = networks.AutoEncoder(
            encoder=encoder,
            decoder=decoder,
        )
        if test_loader is not None:
            with torch.no_grad():
                baseline_loss = 0
                for batch in test_loader:
                    out = autoencoder(batch)
                    baseline_loss += self.TEST_CRITERION(out, batch)
        else:
            baseline_loss = None

        train_losses = autoencoder.fit(train_loader, epochs=epochs)

        if test_loader is not None:
            with torch.no_grad():
                test_loss = 0
                for batch in test_loader:
                    out = autoencoder(batch)
                    test_loss += self.TEST_CRITERION(out, batch)
        else:
            test_loss = None

        artifact = TrainArtifact(
            model=autoencoder,
            train_losses=train_losses,
            baseline_loss=baseline_loss,
            test_loss=test_loss,
        )

        return artifact
