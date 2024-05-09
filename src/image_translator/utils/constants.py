"""Constants for Image Translator."""

from pathlib import Path

COMPETITION_NAME = "gan-getting-started"

class Paths:
    ROOT = Path(__file__).resolve().parents[3]
    ASSETS = ROOT.joinpath("assets")
    IMAGES = ASSETS.joinpath("images")
