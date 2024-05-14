"""Networks module for image translator."""

from typing import Generator, Literal

import torch
from torch import nn

from image_translator.utils.constants import Variables
from image_translator.utils.utils import get_logger


class Encoder(nn.Module):

    def __init__(
        self, latent_dimensions: int = Variables.LATENT_DIMENSIONS, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mainline = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 128 * 128, latent_dimensions),
        )

    def forward(self, x):
        main = self.mainline(x)
        out = self.fc(main)
        return out


class Decoder(nn.Module):
    ADAPTER_SHAPE = (64, 128, 128)

    def __init__(
        self, latent_dimensions: int = Variables.LATENT_DIMENSIONS, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.adapter = nn.Linear(latent_dimensions, 64 * 128 * 128)
        self.mainline = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        adapted = self.adapter(x).view(-1, *self.ADAPTER_SHAPE)
        out = self.mainline(adapted)

        return out


class AutoEncoder:

    def __init__(
        self, encoder: Encoder, decoder: Decoder, device: Literal["cpu", "cuda"] = "cpu"
    ) -> None:
        self.encoder = encoder.to(device=device)
        self.decoder = decoder.to(device=device)
        self.device = device

        self.logger = get_logger(self.__name__())
        self.logger.info("AutoEncoder initialized on device: {}".format(self.device))

    def __name__(self) -> str:
        return "AutoEncoder"

    def parameters(self) -> Generator:
        encoder_parameters = self.encoder.parameters()
        decoder_parameters = self.decoder.parameters()

        for param in encoder_parameters:
            yield param

        for param in decoder_parameters:
            yield param

    def compress(self, original: torch.Tensor) -> torch.Tensor:
        return self.encoder(original.to(self.device))

    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        return self.decoder(compressed.to(self.device))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(original=original))

    def __call__(self, original: torch.Tensor) -> torch.Tensor:
        return self.forward(original)
