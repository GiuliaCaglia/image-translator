"""Fetch training data from kaggle."""

import zipfile

import click
import kaggle

from image_translator.utils.constants import COMPETITION_NAME, Paths


@click.command()
def fetch_data():
    kaggle.api.competition_download_files(COMPETITION_NAME, path=Paths.ASSETS)

    zip_path = next(Paths.ASSETS.glob("*.zip"))

    with zipfile.ZipFile(zip_path, "r") as ref:
        for file_name in ref.namelist():
            ref.extract(file_name, Paths.IMAGES)
