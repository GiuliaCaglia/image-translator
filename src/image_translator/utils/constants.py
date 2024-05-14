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


class Variables:
    SEED = 42
    LATENT_DIMENSIONS: int = 2
