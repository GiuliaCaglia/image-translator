"""Test Module for networks."""

import pytest
import torch
from torch import nn

from image_translator.networks import networks
from image_translator.utils.constants import Variables


class TestNetworks:
    NUM_SAMPLES = 10
    MOCK_IMAGES = torch.randint(0, 256, (NUM_SAMPLES, 3, 256, 256), dtype=torch.float32)
    ENCODED_SHAPE = (NUM_SAMPLES, Variables.LATENT_DIMENSIONS)
    MOCK_ENCODED = torch.randn(ENCODED_SHAPE)

    @pytest.fixture(autouse=True)
    def conv_blocks(self):
        self.encoder_block = networks.ConvBlock(
            in_channels=3,
            num_hidden_layers=2,
            out_channels=6,
            final=nn.MaxPool2d(2),
            padding=1,
        )
        self.decoder_block = networks.ConvBlock(
            in_channels=6,
            out_channels=3,
            num_hidden_layers=2,
            initializer=nn.UpsamplingNearest2d(scale_factor=2),
            padding=1,
        )

    def test_encoder_forward_has_shape(self):
        encoder = networks.Encoder(
            conv_blocks=[self.encoder_block], adapter_shape=(6, 128, 128)
        )

        actual = encoder(self.MOCK_IMAGES)

        assert actual.shape == self.ENCODED_SHAPE

    def test_decoder_forward_has_shape(self):
        decoder = networks.Decoder(
            conv_blocks=[self.decoder_block], adapter_shape=(6, 128, 128)
        )

        actual = decoder(self.MOCK_ENCODED)

        assert actual.shape == self.MOCK_IMAGES.shape
