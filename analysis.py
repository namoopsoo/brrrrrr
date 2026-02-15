"""
Analysis from ChatGPT, 
new archive.json that includes data through 2026-02-14
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# ----------------------------
# 1) Load archive.json -> dataframe
# ----------------------------
json_path = "/mnt/data/archive.json"

with open(json_path, "r") as f:
    data = json.load(f)

daily = data["daily"]

df = pd.DataFrame({
    "date": pd.to_datetime(daily["time"]),
    "low_temp_F": daily["temperature_2m_min"],
    "high_temp_F": daily["temperature_2m_max"],
    "feels_like_low_F": daily["apparent_temperature_min"],
    "feels_like_high_F": daily["apparent_temperature_max"],
    "sunrise": pd.to_datetime(daily["sunrise"]),
    "sunset": pd.to_datetime(daily["sunset"]),
})

df = df.sort_values("date").reset_index(drop=True)

# Save reconstructed dataset
csv_path = "/mnt/data/nyc_daily_weather_2016_02_15_to_2026_02_14.csv"
df.to_csv(csv_path, index=False)

# ----------------------------
# 2) Full 10-year time-series plot
# ----------------------------
plt.figure(figsize=(12,6))

plt.plot(df["date"], df["low_temp_F"])
plt.plot(df["date"], df["high_temp_F"])
plt.plot(df["date"], df["feels_like_low_F"])
plt.plot(df["date"], df["feels_like_high_F"])

plt.title("NYC Daily Temperatures (2016-02-15 → 2026-02-14)")
plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.tight_layout()

ts_plot_path = "/mnt/data/nyc_temperature_time_series_2016_2026.png"
plt.savefig(ts_plot_path, dpi=150)
plt.close()

# ----------------------------
# 3) Rolling 14-day averages (excluding current day)
# ----------------------------
window = 14

df["w14_high_avg_F"] = df["high_temp_F"].rolling(window).mean().shift(1)
df["w14_feels_like_high_avg_F"] = df["feels_like_high_F"].rolling(window).mean().shift(1)

# Ranked lists
rank_high = (
    df.dropna(subset=["w14_high_avg_F"])
      .sort_values("w14_high_avg_F")
      [["date","w14_high_avg_F"]]
)

rank_feels = (
    df.dropna(subset=["w14_feels_like_high_avg_F"])
      .sort_values("w14_feels_like_high_avg_F")
      [["date","w14_feels_like_high_avg_F"]]
)

rank_high_path = "/mnt/data/nyc_ranked_coldest_14day_high_avg.csv"
rank_feels_path = "/mnt/data/nyc_ranked_coldest_14day_feels_like_high_avg.csv"

rank_high.to_csv(rank_high_path, index=False)
rank_feels.to_csv(rank_feels_path, index=False)

# ----------------------------
# 4) Build BEAUTIFUL stacked plot for top 5 coldest years
# ----------------------------
df["year"] = df["date"].dt.year

yearly_min = (
    df.dropna(subset=["w14_high_avg_F"])
      .sort_values("w14_high_avg_F")
      .groupby("year")
      .first()
      .reset_index()
)

top5_years = yearly_min.sort_values("w14_high_avg_F").head(5)["year"].tolist()

rows = []
meta = []

for y in top5_years:
    row = yearly_min[yearly_min["year"] == y].iloc[0]
    end_date_exclusive = row["date"]
    start_date = end_date_exclusive - pd.Timedelta(days=14)
    end_date_inclusive = end_date_exclusive - pd.Timedelta(days=1)
    
    mask = (df["date"] >= start_date) & (df["date"] < end_date_exclusive)
    window_df = df.loc[mask, ["date","high_temp_F"]].copy()
    window_df["year"] = y
    window_df["day_index"] = range(len(window_df))
    rows.append(window_df)
    
    meta.append({
        "year": y,
        "start": start_date,
        "end_inclusive": end_date_inclusive,
        "avg_high": window_df["high_temp_F"].mean()
    })

plot_df = pd.concat(rows)
meta_df = pd.DataFrame(meta)

# Plot stacked
fig, ax = plt.subplots(figsize=(12,6.2))
offset_step = 40
blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

for i, y in enumerate(sorted(top5_years)):
    sub = plot_df[plot_df["year"] == y]
    offset = i * offset_step
    
    series = sub["high_temp_F"] - sub["high_temp_F"].mean() + offset
    ax.plot(sub["day_index"], series, linewidth=2)
    
    start_x, start_y = sub["day_index"].iloc[0], series.iloc[0]
    end_x, end_y = sub["day_index"].iloc[-1], series.iloc[-1]
    start_temp = round(sub["high_temp_F"].iloc[0])
    end_temp = round(sub["high_temp_F"].iloc[-1])
    
    ax.scatter(start_x, start_y)
    ax.text(start_x-0.3, start_y+1, f"{start_temp}", ha="right")
    ax.scatter(end_x, end_y)
    ax.text(end_x+0.2, end_y+1, f"{end_temp}", ha="left")
    
    m = meta_df[meta_df["year"] == y].iloc[0]
    date_label = f"{m['start']:%b} {m['start'].day}, {m['start']:%Y}–{m['end_inclusive']:%b} {m['end_inclusive'].day}, {m['end_inclusive']:%Y}"
    ax.text(-0.30, offset, date_label, transform=blend, va="center", ha="left")
    
    ax.text(6.5, offset, f"{m['avg_high']:.1f}°F avg",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0))

ax.set_yticks([])
ax.set_xlabel("Day within 14-day window")
ax.set_title("Coldest 14-Day Windows by Year (Top 5 Coldest Years)")

fig.subplots_adjust(left=0.34)

stacked_plot_path = "/mnt/data/nyc_coldest_windows_stacked_plot_labeled_dates_avg_2026.png"
plt.savefig(stacked_plot_path, dpi=150)
plt.close()

csv_path, ts_plot_path, rank_high_path, rank_feels_path, stacked_plot_path
