"""
Phase 3 — cleaning and standardizing the validated raw price and news data.
Does NOT do sentiment scoring (Phase 5), merging (Phase 7), feature
engineering (Phase 8), or target generation (Phase 9) — outputs go to
data/interim/ as clean, filtered, standalone series.
"""

import re
from pathlib import Path
import pandas as pd

# --- paths — using the ACTUAL confirmed locations from validation, not the
# original data/raw/ convention for price data specifically. Price data
# currently sits outside data/raw/; consider moving it to data/raw/prices/
# for consistency with news, but not required for this to run. ---
PRICE_PARQUET_DIR = Path(r"C:\Users\siddh\Desktop\PBL\Data\parquet")
NEWS_CSV_PATH = Path(r"C:\Users\siddh\Desktop\PBL\data\raw\news\india-news-headlines.csv")
INTERIM_DIR = Path(r"C:\Users\siddh\Desktop\PBL\data\interim")

TARGET_COMMODITIES = ["Onion", "Wheat", "Soyabean"]  # confirmed exact spellings — NOT "Soybean"
BENCHMARK_MARKETS = {
    "Onion": "Lasalgaon",
    "Wheat": "Kanpur (Grain)",
    "Soyabean": "Indore",
}

MODELING_START = "2013-07-01"  # corrected: verified soyabean price start, not 2012-06-01
MODELING_END = "2023-06-30"    # verified news dataset end

COMMODITY_PATTERNS = {
    "Onion": r"\bonion\b",
    "Wheat": r"\bwheat\b",
    "Soyabean": r"\b(soybean|soyabean)\b",  # both spellings — confirmed both occur in news
}
EXCLUDED_NEWS_CATEGORY = "business.international-business"  # confirmed: global wire content, not India-specific


def load_price_data(parquet_dir: Path) -> pd.DataFrame:
    """Load yearly Agmarknet parquet files, filter to target commodities and
    benchmark markets only, and collapse multiple varieties/day into one
    representative price per (commodity, date)."""
    year_files = sorted(parquet_dir.glob("*.parquet"))
    if not year_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    frames = [pd.read_parquet(f, filters=[("Commodity", "in", TARGET_COMMODITIES)])
              for f in year_files]
    df = pd.concat(frames, ignore_index=True)

    df = df.drop_duplicates().reset_index(drop=True)  # 266 confirmed exact duplicates

    # normalize the Nov-2025 "APMC" naming change — currently a no-op given our
    # window ends before that rename, kept for robustness if the window changes later
    df["base_market"] = df["Market"].str.replace(r"\s*APMC$", "", regex=True).str.strip()

    kept = [df[(df["Commodity"] == c) & (df["base_market"] == m)]
            for c, m in BENCHMARK_MARKETS.items()]
    df = pd.concat(kept, ignore_index=True)

    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

    daily = (
        df.groupby(["Commodity", "Arrival_Date"], as_index=False)
          .agg(price=("Modal_Price", "mean"),
               min_price=("Min_Price", "mean"),
               max_price=("Max_Price", "mean"),
               n_varieties_reported=("Variety", "nunique"))
    )
    return daily


def load_and_tag_news(csv_path: Path) -> pd.DataFrame:
    """Load headlines, tag each with which target commodity it mentions
    (word-boundary matched), exclude international-wire noise. An article
    mentioning multiple commodities produces one row per match (deliberate —
    see Architecture Check point 2)."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)  # 25,645 confirmed exact duplicates
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d", errors="coerce")

    tagged_frames = []
    for commodity, pattern in COMMODITY_PATTERNS.items():
        mask = df["headline_text"].str.contains(pattern, case=False, na=False, regex=True)
        subset = df[mask].copy()
        subset["commodity"] = commodity
        tagged_frames.append(subset)
    tagged = pd.concat(tagged_frames, ignore_index=True)

    before = len(tagged)
    tagged = tagged[tagged["headline_category"] != EXCLUDED_NEWS_CATEGORY].reset_index(drop=True)
    print(f"Excluded {before - len(tagged)} international-business wire headlines "
          f"({before} -> {len(tagged)})")

    return tagged


def apply_modeling_window(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    start, end = pd.Timestamp(MODELING_START), pd.Timestamp(MODELING_END)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].reset_index(drop=True)


if __name__ == "__main__":
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_price_data(PRICE_PARQUET_DIR)
    prices = apply_modeling_window(prices, "Arrival_Date")
    prices.to_csv(INTERIM_DIR / "prices_clean.csv", index=False)
    print("\n=== Prices ===")
    print(prices.shape)
    print(prices.groupby("Commodity").agg(
        rows=("price", "size"),
        start=("Arrival_Date", "min"),
        end=("Arrival_Date", "max"),
        missing_price=("price", lambda s: s.isnull().sum())
    ))

    news = load_and_tag_news(NEWS_CSV_PATH)
    news = apply_modeling_window(news, "publish_date")
    news.to_csv(INTERIM_DIR / "news_clean.csv", index=False)
    print("\n=== News ===")
    print(news.shape)
    print(news.groupby("commodity").agg(
        articles=("headline_text", "size"),
        unique_dates=("publish_date", "nunique"),
        start=("publish_date", "min"),
        end=("publish_date", "max")
    ))