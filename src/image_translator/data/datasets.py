"""Dataset classes for image-translator."""

from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from image_translator.utils.constants import Paths, Variables


class TrainTestSplitPaths:
    SEED = Variables.SEED
    IMAGE_PATHS = Paths.IMAGES.glob("*/*.jpg")

    @classmethod
    def get_split(cls, train_size: float) -> Tuple[List[Path], List[Path]]:
        rng = np.random.default_rng(cls.SEED)
        paths = list(cls.IMAGE_PATHS)
        num_train_samples = floor(len(paths) * train_size)
        train_paths = rng.choice(paths, size=num_train_samples, replace=False).tolist()
        test_paths = [path for path in paths if path not in train_paths]

        return train_paths, test_paths


class ImageDataset(Dataset):
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.tensor:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        return self.TRANSFORM(image)
