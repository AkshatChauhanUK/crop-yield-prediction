"""
eda.py
------
Exploratory Data Analysis on cleaned crop yield data.
Generates summary stats + saves charts as PNG files in the outputs/ folder.

Run:
    python src/eda.py --input data/cleaned_crop_data.csv --outdir outputs
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_summary(df: pd.DataFrame):
    print("=" * 50)
    print("BASIC INFO")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("Columns:", list(df.columns), "\n")
    print("Data types:")
    print(df.dtypes, "\n")
    print("Summary statistics (numeric columns):")
    print(df.describe().round(2), "\n")
    print(f"Unique crops: {df['crop'].nunique()} -> {sorted(df['crop'].unique())}\n")
    print(f"Unique states: {df['state'].nunique()} -> {sorted(df['state'].unique())}\n")


def plot_yield_by_crop(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(10, 6))
    order = df.groupby("crop")["yield_quintal_hectare"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="crop", y="yield_quintal_hectare", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Yield Distribution by Crop")
    plt.ylabel("Yield (Quintal/Hectare)")
    plt.xlabel("Crop")
    plt.tight_layout()
    path = os.path.join(outdir, "yield_by_crop.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_yield_by_state(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(10, 6))
    avg_yield = df.groupby("state")["yield_quintal_hectare"].mean().sort_values(ascending=False)
    sns.barplot(x=avg_yield.values, y=avg_yield.index, palette="viridis")
    plt.title("Average Yield by State")
    plt.xlabel("Average Yield (Quintal/Hectare)")
    plt.ylabel("State")
    plt.tight_layout()
    path = os.path.join(outdir, "yield_by_state.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_cost_vs_yield(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="cost_of_cultivation_`_hectare_c2",
        y="yield_quintal_hectare",
        hue="crop",
        s=80,
    )
    plt.title("Cost of Cultivation vs Yield")
    plt.xlabel("Cost of Cultivation (Rs/Hectare, C2)")
    plt.ylabel("Yield (Quintal/Hectare)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(outdir, "cost_vs_yield.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, outdir: str):
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(7, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    path = os.path.join(outdir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_yield_by_crop_log(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(10, 6))
    order = df.groupby("crop")["yield_quintal_hectare"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="crop", y="yield_quintal_hectare", order=order)
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.title("Yield Distribution by Crop (Log Scale)")
    plt.ylabel("Yield (Quintal/Hectare) - Log Scale")
    plt.xlabel("Crop")
    plt.tight_layout()
    path = os.path.join(outdir, "yield_by_crop_log.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def detect_outliers(df: pd.DataFrame):
    print("=" * 50)
    print("POTENTIAL OUTLIERS (yield > mean + 2*std, per crop)")
    print("=" * 50)
    for crop, group in df.groupby("crop"):
        mean = group["yield_quintal_hectare"].mean()
        std = group["yield_quintal_hectare"].std()
        if pd.isna(std) or std == 0:
            continue
        outliers = group[
            (group["yield_quintal_hectare"] > mean + 2 * std)
            | (group["yield_quintal_hectare"] < mean - 2 * std)
        ]
        if not outliers.empty:
            print(f"\n{crop}:")
            print(outliers[["state", "yield_quintal_hectare"]].to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cleaned_crop_data.csv")
    parser.add_argument("--outdir", required=True, help="Folder to save chart PNGs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)

    basic_summary(df)
    detect_outliers(df)

    plot_yield_by_crop(df, args.outdir)
    plot_yield_by_crop_log(df, args.outdir)
    plot_yield_by_state(df, args.outdir)
    plot_cost_vs_yield(df, args.outdir)
    plot_correlation_heatmap(df, args.outdir)

    print("EDA complete. Check the outputs folder for charts.")


if __name__ == "__main__":
    main()