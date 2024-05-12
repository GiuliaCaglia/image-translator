"""Dataset classes for image-translator."""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from image_translator.utils.constants import Paths


class ImageDataset(Dataset):
    IMAGE_PATHS = Paths.IMAGES.glob("*.jpg")

    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self) -> None:
        self.paths = list(self.IMAGE_PATHS)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.tensor:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        return self.TRANSFORM(image)
