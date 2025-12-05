import pandas as pd
from pathlib import Path

"""
Util script to enrich ad copy csv to include:
    - ctr (click-through rate)
    - cvr (conversion rate)
    - cpc (cost per click)
    - cpa (cost per aquisition)
    - high_performer (binary label for modeling/LLM conditioning)
"""

# --------- CONFIG ----------
INPUT_CSV = "../../data/enriched_ads_7k_complete.csv"      # path to your raw dataset
OUTPUT_CSV = "../../data/enriched_ads_with_metrics.csv"  # path to save enriched dataset
# ---------------------------


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(p)
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ctr = clicks / impressions
      - cvr = conversions / clicks
      - cpc = spend / clicks
      - cpa = spend / conversions
      - high_performer = 1 if ctr >= ctr_percentile_threshold else 0
    """

    # Avoid division by 0 with safe denominators
    df["impressions"] = df["impressions"].fillna(0)
    df["clicks"] = df["clicks"].fillna(0)
    df["conversions"] = df["conversions"].fillna(0)
    df["spend"] = df["spend"].fillna(0.0)

    # CTR: clicks / impressions
    df["ctr"] = df.apply(
        lambda row: row["clicks"] / row["impressions"] if row["impressions"] > 0 else 0.0,
        axis=1,
    )

    # CVR: conversions / clicks
    df["cvr"] = df.apply(
        lambda row: row["conversions"] / row["clicks"] if row["clicks"] > 0 else 0.0,
        axis=1,
    )

    # CPC: spend / clicks
    df["cpc"] = df.apply(
        lambda row: row["spend"] / row["clicks"] if row["clicks"] > 0 else 0.0,
        axis=1,
    )

    # CPA: spend / conversions
    df["cpa"] = df.apply(
        lambda row: row["spend"] / row["conversions"] if row["conversions"] > 0 else 0.0,
        axis=1,
    )

    # High performer flag using a global CTR percentile (e.g., 75th)
    ctr_threshold = df["ctr"].quantile(0.75)
    df["high_performer"] = (df["ctr"] >= ctr_threshold).astype(int)

    return df


def main():
    df = load_data(INPUT_CSV)
    df_enriched = add_derived_metrics(df)
    df_enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved enriched dataset with derived metrics to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
