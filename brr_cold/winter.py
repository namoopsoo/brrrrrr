from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# ------------------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherData:
    df: pl.DataFrame


# ------------------------------------------------------------------------------
# Loading Open-Meteo archive.json
# ------------------------------------------------------------------------------

def load_open_meteo_archive_json(path: str | Path) -> WeatherData:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    daily = data["daily"]

    df = pl.DataFrame({
        "date": pl.Series(daily["time"]).str.strptime(pl.Date),
        "low_temp_F": daily["temperature_2m_min"],
        "high_temp_F": daily["temperature_2m_max"],
        "feels_like_low_F": daily["apparent_temperature_min"],
        "feels_like_high_F": daily["apparent_temperature_max"],
        "sunrise": pl.Series(daily["sunrise"]).str.strptime(pl.Datetime),
        "sunset": pl.Series(daily["sunset"]).str.strptime(pl.Datetime),
    }).sort("date")

    return WeatherData(df=df)


# ------------------------------------------------------------------------------
# Rolling windows (exclude current day)
# ------------------------------------------------------------------------------

def add_rolling_14day_averages_excluding_today(df: pl.DataFrame) -> pl.DataFrame:
    """
    w14_high_avg_F(t) =
        mean(high_temp_F(t-14 ... t-1))
    """

    return df.with_columns([
        pl.col("high_temp_F")
            .rolling_mean(window_size=14)
            .shift(1)
            .alias("w14_high_avg_F"),

        pl.col("feels_like_high_F")
            .rolling_mean(window_size=14)
            .shift(1)
            .alias("w14_feels_like_high_avg_F"),
    ])


# ------------------------------------------------------------------------------
# Ranked coldest windows
# ------------------------------------------------------------------------------

def ranked_coldest_14day_averages(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (
        df.drop_nulls(col)
          .select(["date", col])
          .sort(col)
    )


# ------------------------------------------------------------------------------
# Full time-series plot
# ------------------------------------------------------------------------------

def plot_full_timeseries(df: pl.DataFrame, title: str = "NYC Daily Temperatures") -> None:

    pdf = df.to_pandas()  # matplotlib expects array-like

    plt.figure(figsize=(12, 6))
    plt.plot(pdf["date"], pdf["low_temp_F"])
    plt.plot(pdf["date"], pdf["high_temp_F"])
    plt.plot(pdf["date"], pdf["feels_like_low_F"])
    plt.plot(pdf["date"], pdf["feels_like_high_F"])

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Temperature (°F)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Dual stacked plot (top 5 coldest years)
# ------------------------------------------------------------------------------

def dual_stacked_plot_top5_coldest_years(
    df_with_roll: pl.DataFrame,
    metric_col: str = "w14_high_avg_F",
    title: str = "Dual Comparison: High Temp vs Feels-Like High (Top 5 Coldest Years)",
) -> None:

    df = df_with_roll.with_columns(
        pl.col("date").dt.year().alias("year")
    )

    yearly_min = (
        df.drop_nulls(metric_col)
          .sort(metric_col)
          .group_by("year")
          .first()
    )

    top5_years = (
        yearly_min
        .sort(metric_col)
        .select("year")
        .head(5)
        .to_series()
        .to_list()
    )

    fig, ax = plt.subplots(figsize=(12, 6.2))
    offset_step = 45
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, y in enumerate(sorted(top5_years)):
        row = yearly_min.filter(pl.col("year") == y).to_dicts()[0]

        end_date = row["date"]
        start_date = end_date - pl.duration(days=14)

        window_df = (
            df.filter(
                (pl.col("date") >= start_date)
                & (pl.col("date") < end_date)
            )
            .select(["date", "high_temp_F", "feels_like_high_F"])
            .with_row_count("day_index")
        )

        offset = i * offset_step

        high_mean = window_df["high_temp_F"].mean()
        feels_mean = window_df["feels_like_high_F"].mean()

        series_high = window_df["high_temp_F"] - high_mean + offset
        series_feels = window_df["feels_like_high_F"] - feels_mean + offset

        ax.plot(window_df["day_index"], series_high, linewidth=2)
        ax.plot(window_df["day_index"], series_feels, linewidth=1)

        date_label = (
            f"{start_date.strftime('%b')} {start_date.day}, {start_date.year}"
            f"–{(end_date - pl.duration(days=1)).strftime('%b')} "
            f"{(end_date - pl.duration(days=1)).day}, {(end_date - pl.duration(days=1)).year}"
        )

        ax.text(-0.30, offset, date_label, transform=blend,
                va="center", ha="left")

        ax.text(
            6.5, offset,
            f"{high_mean:.1f}°F avg (high)\n{feels_mean:.1f}°F avg (feels)",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white",
                      alpha=0.75,
                      linewidth=0),
        )

    ax.set_yticks([])
    ax.set_xlabel("Day within 14-day window")
    ax.set_title(title)
    fig.subplots_adjust(left=0.34)

    plt.show()
