"""Utility functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Type

import yaml
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

CONFIG_ELEMENTS = {
    "conv": nn.Conv2d,
    "batchnorm": nn.BatchNorm2d,
    "relu6": nn.ReLU6,
    "linear": nn.Linear,
    "flatten": nn.Flatten,
    "sigmoid": nn.Sigmoid,
    "unflatten": nn.Unflatten,
    "transpose_conv": nn.ConvTranspose2d,
}


@dataclass
class TrainingParams:
    optimizer: Type[Optimizer]
    epochs: int
    loss_function: nn.Module
    learning_rate: float
    batch_size: int
    smoke_test: bool
    device: Literal["cpu", "cuda"] = "cpu"

    __CONVERTER = {"Adam": Adam, "mse_loss": nn.MSELoss()}

    def __post_init__(self):
        self.optimizer = self.__CONVERTER[self.optimizer]
        self.loss_function = self.__CONVERTER[self.loss_function]
        self.learning_rate = float(self.learning_rate)

    @classmethod
    def load_yaml(cls, path: Path) -> TrainingParams:
        with path.open("r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="%(asctime)s %(name)s - %(message)s", level=logging.INFO)

    return logging.getLogger(name=name)
