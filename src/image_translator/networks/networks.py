"""Networks module for image translator."""

from torch import nn

from image_translator.utils.constants import Variables


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
