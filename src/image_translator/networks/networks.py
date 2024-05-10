"""Networks module for image translator."""

from torch import nn


class Encoder(nn.Module):

    def __init__(self, latent_dimensions: int = 2):
        super().__init__()
        self.mainline = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 128 * 128, latent_dimensions),
            )

    def forward(self, x):
        main = self.mainline(x)
        out = self.fc(main)
        return out
