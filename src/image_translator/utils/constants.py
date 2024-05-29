"""Constants for Image Translator."""

from pathlib import Path

COMPETITION_NAME = "gan-getting-started"


class Paths:
    ROOT = Path(__file__).resolve().parents[3]
    ASSETS = ROOT.joinpath("assets")
    IMAGES = ASSETS.joinpath("images")
    RESULTS = ASSETS.joinpath("results")

    MODEL = RESULTS.joinpath("model.pkl")
    METRICS = RESULTS.joinpath("metrics.json")
    TRAIN_SAMPLES = RESULTS.joinpath("train_samples.jpg")
    TEST_SAMPLES = RESULTS.joinpath("test_samples.jpg")
    LOSS_PLOT = RESULTS.joinpath("loss_plot.png")

    CONFIG = ASSETS / "config"
    ENCODER_CONFIG = CONFIG / "encoder_config.yaml"
    DECODER_CONFIG = CONFIG / "decoder_config.yaml"
    TRAIN_CONFIG = CONFIG / "training_config.yaml"


class Variables:
    SEED = 42


TYPE = "type"
PARAMS = "params"
