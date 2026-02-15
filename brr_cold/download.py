
"""
Download the initial data from open-meteo.com
"""
import json
import requests
from pathlib import Path


def download_open_meteo():
    url = """https://archive-api.open-meteo.com/v1/archive?
    latitude=40.7128
    &longitude=-74.0060
    &start_date=2016-02-15
    &end_date=2026-02-14
    &daily=temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset
    &temperature_unit=fahrenheit
    &timezone=America/New_York""".replace("\n", "")

    response = requests.get(url)

    if str(response.status_code).startswith("2"):
        Path("data/archive.json").write_text(
            json.dumps(response.json(), indent=5)
        )
