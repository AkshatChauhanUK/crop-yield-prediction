"""
Time Series Analysis — Crop Yield Trends in India
Analyzes year-over-year yield trends from the crop_production dataset
(State_Name, District_Name, Crop_Year, Season, Crop, Area, Production, Yield).

Run: python src/time_series.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

OUTPUT_DIR = "outputs"
DATA_PATH = "data/crop_production.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load & Clean ──────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename for consistency
df = df.rename(columns={
    "state_name": "state",
    "district_name": "district",
    "crop_year": "year",
    "crop": "crop",
    "area": "area",
    "production": "production",
})

# Drop rows with missing yield/production/area
df = df.dropna(subset=["production", "area"])
df = df[df["area"] > 0]

# Compute yield if not present (production / area)
if "yield" not in df.columns:
    df["yield"] = df["production"] / df["area"]

df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
df = df.dropna(subset=["yield"])
df = df[df["yield"] > 0]

# Clean year column
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

print(f"Clean shape: {df.shape}")
print(f"Year range: {df['year'].min()} — {df['year'].max()}")
print(f"Unique crops: {df['crop'].nunique()}")
print(f"Unique states: {df['state'].nunique()}")

# ── 2. National Year-over-Year Yield Trend (Top 10 Crops) ────────────────────

# Top 10 crops by average yield
top_crops = (
    df.groupby("crop")["yield"]
    .mean()
    .nlargest(10)
    .index.tolist()
)

national_trend = (
    df[df["crop"].isin(top_crops)]
    .groupby(["year", "crop"])["yield"]
    .mean()
    .reset_index()
)

fig1 = px.line(
    national_trend,
    x="year", y="yield", color="crop",
    title="Year-over-Year Yield Trends — Top 10 Crops (National Average)",
    labels={"year": "Year", "yield": "Yield (Kg/Hectare)", "crop": "Crop"},
    markers=True,
)
fig1.update_layout(
    hovermode="x unified",
    legend_title="Crop",
    template="plotly_dark",
)
fig1.write_html(f"{OUTPUT_DIR}/timeseries_national_top10.html")
print("Saved: timeseries_national_top10.html")

# ── 3. State-wise Trend for Key Crops ────────────────────────────────────────

KEY_CROPS = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton"]
KEY_CROPS = [c for c in KEY_CROPS if c in df["crop"].unique()]

# Top 5 states by total production for each key crop
state_trend_frames = []
for crop in KEY_CROPS:
    crop_df = df[df["crop"] == crop]
    top_states = (
        crop_df.groupby("state")["production"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )
    trend = (
        crop_df[crop_df["state"].isin(top_states)]
        .groupby(["year", "state"])["yield"]
        .mean()
        .reset_index()
    )
    trend["crop"] = crop
    state_trend_frames.append(trend)

state_trend = pd.concat(state_trend_frames, ignore_index=True)

fig2 = px.line(
    state_trend,
    x="year", y="yield", color="state",
    facet_col="crop", facet_col_wrap=3,
    title="State-wise Yield Trends — Key Crops (Top 5 States per Crop)",
    labels={"year": "Year", "yield": "Yield (Kg/Hectare)", "state": "State"},
    markers=True,
)
fig2.update_layout(template="plotly_dark", height=700)
fig2.write_html(f"{OUTPUT_DIR}/timeseries_statewise.html")
print("Saved: timeseries_statewise.html")

# ── 4. Growth Rate Analysis ──────────────────────────────────────────────────

# Average yield per year per crop — national
pivot = (
    df[df["crop"].isin(top_crops)]
    .groupby(["year", "crop"])["yield"]
    .mean()
    .unstack("crop")
)

# Year-over-year % change
growth = pivot.pct_change() * 100
growth = growth.reset_index().melt(id_vars="year", var_name="crop", value_name="growth_pct")
growth = growth.dropna()

fig3 = px.bar(
    growth,
    x="year", y="growth_pct", color="crop",
    barmode="group",
    title="Year-over-Year Yield Growth Rate (%) — Top 10 Crops",
    labels={"year": "Year", "growth_pct": "YoY Growth (%)", "crop": "Crop"},
)
fig3.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
fig3.update_layout(template="plotly_dark", height=500)
fig3.write_html(f"{OUTPUT_DIR}/timeseries_growth_rate.html")
print("Saved: timeseries_growth_rate.html")

# ── 5. Heatmap — State × Year for Rice ───────────────────────────────────────

rice_df = df[df["crop"] == "Rice"] if "Rice" in df["crop"].unique() else df[df["crop"] == df["crop"].iloc[0]]
rice_pivot = (
    rice_df.groupby(["state", "year"])["yield"]
    .mean()
    .unstack("year")
)
# Keep only states with enough data
rice_pivot = rice_pivot.dropna(thresh=10)

fig4 = px.imshow(
    rice_pivot,
    title="Rice Yield Heatmap — State × Year (Kg/Hectare)",
    labels={"x": "Year", "y": "State", "color": "Yield"},
    color_continuous_scale="YlGn",
    aspect="auto",
)
fig4.update_layout(template="plotly_dark", height=600)
fig4.write_html(f"{OUTPUT_DIR}/timeseries_heatmap_rice.html")
print("Saved: timeseries_heatmap_rice.html")

# ── 6. Summary Stats ─────────────────────────────────────────────────────────

summary = (
    df.groupby("crop")["yield"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
    .head(20)
)
summary.to_csv(f"{OUTPUT_DIR}/timeseries_summary.csv")
print("Saved: timeseries_summary.csv")

print("\n✅ Time series analysis complete!")
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("\nFiles generated:")
print("  - timeseries_national_top10.html  (national trend, top 10 crops)")
print("  - timeseries_statewise.html       (state-wise trends, key crops)")
print("  - timeseries_growth_rate.html     (YoY growth rate chart)")
print("  - timeseries_heatmap_rice.html    (rice yield heatmap)")
print("  - timeseries_summary.csv          (summary statistics)")