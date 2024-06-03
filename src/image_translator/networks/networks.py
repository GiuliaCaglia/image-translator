"""Networks module for image translator."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Literal

import torch
import yaml
from torch import nn

from image_translator.utils.constants import PARAMS, TYPE
from image_translator.utils.utils import CONFIG_ELEMENTS, get_logger


class Coder(nn.Module):

    @classmethod
    def from_config(cls, path: Path) -> Coder:
        # Open Yaml
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        sequence = []
        for layer in config:
            next_layer = CONFIG_ELEMENTS[layer[TYPE]](**layer.get(PARAMS, {}))
            sequence.append(next_layer)

        return cls(sequence)

    def __init__(self, modules: List[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainline = nn.Sequential(*modules)

    def forward(self, x):
        main = self.mainline(x)
        return main


class AutoEncoder(nn.Module):

    def __init__(
        self, encoder: Coder, decoder: Coder, device: Literal["cpu", "cuda"] = "cpu"
    ) -> None:
        super().__init__()
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
