"""
EDA for hotels_sample.csv

Run with:  .venv/bin/python hotels.py
"""

import pandas as pd

df = pd.read_csv("hotels_sample.csv")

SEP = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Shape & types ──────────────────────────────────────────
section("1. SHAPE & DTYPES")
print(f"Rows : {df.shape[0]}")
print(f"Cols : {df.shape[1]}")
print()
print(df.dtypes)

# ── 2. First / last rows ──────────────────────────────────────
section("2. FIRST 5 ROWS")
print(df.head().to_string(index=False))

section("3. LAST 5 ROWS")
print(df.tail().to_string(index=False))

# ── 3. Missing values ─────────────────────────────────────────
section("4. MISSING VALUES")
missing = df.isnull().sum()
pct = (missing / len(df) * 100).round(2)
mv = pd.DataFrame({"missing": missing, "pct_%": pct})
print(mv[mv["missing"] > 0].to_string() if mv["missing"].any() else "No missing values.")

# ── 4. Numeric summary ────────────────────────────────────────
section("5. NUMERIC SUMMARY")
print(df.describe(include="number").T.to_string())

# ── 5. avg_score distribution ────────────────────────────────
section("6. AVG_SCORE DISTRIBUTION")
bins = [0, 5, 6, 7, 7.5, 8, 8.5, 9, 10]
labels = ["≤5", "5-6", "6-7", "7-7.5", "7.5-8", "8-8.5", "8.5-9", "9-10"]
df["score_band"] = pd.cut(df["avg_score"], bins=bins, labels=labels, right=True)
print(df["score_band"].value_counts().sort_index().to_string())
df.drop(columns="score_band", inplace=True)

print(f"\nMean   : {df['avg_score'].mean():.2f}")
print(f"Median : {df['avg_score'].median():.2f}")
print(f"Std    : {df['avg_score'].std():.2f}")
print(f"Min    : {df['avg_score'].min():.1f}")
print(f"Max    : {df['avg_score'].max():.1f}")

# ── 6. total_reviews distribution ────────────────────────────
section("7. TOTAL_REVIEWS DISTRIBUTION")
rev_bins = [0, 100, 500, 1000, 2000, 5000, df["total_reviews"].max() + 1]
rev_labels = ["1-100", "101-500", "501-1k", "1k-2k", "2k-5k", "5k+"]
df["rev_band"] = pd.cut(df["total_reviews"], bins=rev_bins, labels=rev_labels, right=True)
print(df["rev_band"].value_counts().sort_index().to_string())
df.drop(columns="rev_band", inplace=True)

print(f"\nMean   : {df['total_reviews'].mean():.0f}")
print(f"Median : {df['total_reviews'].median():.0f}")
print(f"Max    : {df['total_reviews'].max()}")

# ── 7. City breakdown ────────────────────────────────────────
section("8. HOTELS PER CITY")
city_counts = df["city"].value_counts()
print(city_counts.to_string())
print(f"\nUnique cities : {df['city'].nunique()}")

# ── 8. Country breakdown ─────────────────────────────────────
section("9. HOTELS PER COUNTRY")
print(df["country"].value_counts().to_string())

# ── 9. Avg score by city ──────────────────────────────────────
section("10. AVG SCORE BY CITY")
city_stats = (
    df.groupby("city")["avg_score"]
    .agg(count="count", mean="mean", median="median", min="min", max="max")
    .sort_values("mean", ascending=False)
    .round(2)
)
print(city_stats.to_string())

# ── 10. Avg score by country ──────────────────────────────────
section("11. AVG SCORE BY COUNTRY")
country_stats = (
    df.groupby("country")["avg_score"]
    .agg(count="count", mean="mean", median="median", min="min", max="max")
    .sort_values("mean", ascending=False)
    .round(2)
)
print(country_stats.to_string())

# ── 11. Top / bottom hotels ───────────────────────────────────
section("12. TOP 10 HOTELS BY AVG SCORE")
top = df.nlargest(10, "avg_score")[["name", "city", "avg_score", "total_reviews"]]
print(top.to_string(index=False))

section("13. BOTTOM 10 HOTELS BY AVG SCORE")
bot = df.nsmallest(10, "avg_score")[["name", "city", "avg_score", "total_reviews"]]
print(bot.to_string(index=False))

# ── 12. Geo bounds ────────────────────────────────────────────
section("14. GEOGRAPHIC BOUNDS")
print(f"Latitude  : {df['latitude'].min():.4f}  →  {df['latitude'].max():.4f}")
print(f"Longitude : {df['longitude'].min():.4f}  →  {df['longitude'].max():.4f}")

section("15. GEO CENTRE PER CITY")
geo = (
    df.groupby("city")[["latitude", "longitude"]]
    .mean()
    .round(4)
)
print(geo.to_string())

# ── 13. Duplicate checks ──────────────────────────────────────
section("16. DUPLICATE CHECKS")
dup_names = df["name"].duplicated().sum()
dup_coords = df.duplicated(subset=["latitude", "longitude"]).sum()
print(f"Duplicate hotel names      : {dup_names}")
print(f"Duplicate (lat, lng) pairs : {dup_coords}")

# ── 14. Correlation ───────────────────────────────────────────
section("17. CORRELATION (numeric columns)")
print(df[["avg_score", "total_reviews", "latitude", "longitude"]].corr().round(3).to_string())
