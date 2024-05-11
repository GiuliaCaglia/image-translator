"""Test Module for networks."""

import torch

from image_translator.networks import networks
from image_translator.utils.constants import Variables


class TestNetworks:
    NUM_SAMPLES = 10
    MOCK_IMAGES = torch.randint(0, 256, (NUM_SAMPLES, 3, 256, 256), dtype=torch.float32)
    ENCODED_SHAPE = (NUM_SAMPLES, Variables.LATENT_DIMENSIONS)
    MOCK_ENCODED = torch.randn(ENCODED_SHAPE)

    def test_encoder_forward_has_shape(self):
        encoder = networks.Encoder()

        actual = encoder(self.MOCK_IMAGES)

        assert actual.shape == self.ENCODED_SHAPE

    def test_decoder_forward_has_shape(self):
        decoder = networks.Decoder()

        actual = decoder(self.MOCK_ENCODED)

        assert actual.shape == self.MOCK_IMAGES.shape
