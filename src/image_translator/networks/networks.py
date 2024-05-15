"""Networks module for image translator."""

from functools import reduce
from typing import Generator, List, Literal, Tuple

import torch
from torch import nn

from image_translator.utils.constants import Variables
from image_translator.utils.utils import get_logger


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hidden_layers: int,
        out_channels: int,
        initializer: nn.Module = nn.Identity(),
        final: nn.Module = nn.Identity(),
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__()
        input_layer = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                **kwargs,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        hidden_layers: List[nn.Module] = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    **kwargs,
                )
            )
            hidden_layers.append(nn.BatchNorm2d(out_channels))
            hidden_layers.append(nn.ReLU())

        self.mainline = nn.Sequential(initializer, *input_layer, *hidden_layers, final)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        return self.mainline(out)


class Encoder(nn.Module):

    def __init__(
        self,
        conv_blocks: List[ConvBlock],
        adapter_shape: Tuple[int, int, int],
        latent_dimensions: int = Variables.LATENT_DIMENSIONS,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mainline = nn.Sequential(*conv_blocks)
        adapter_size = reduce(lambda a, b: a * b, adapter_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=adapter_size, out_features=latent_dimensions),
        )

    def forward(self, x):
        main = self.mainline(x)
        out = self.fc(main)
        return out


class Decoder(nn.Module):

    def __init__(
        self,
        conv_blocks: List[ConvBlock],
        adapter_shape: Tuple[int, int, int],
        latent_dimensions: int = Variables.LATENT_DIMENSIONS,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.adapter_shape = adapter_shape
        adapter_size = reduce(lambda a, b: a * b, adapter_shape)
        self.adapter = nn.Linear(latent_dimensions, adapter_size)
        self.mainline = nn.Sequential(
            *conv_blocks,
            nn.Sigmoid(),
        )

    def forward(self, x):
        adapted = self.adapter(x).view(-1, *self.adapter_shape)
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
