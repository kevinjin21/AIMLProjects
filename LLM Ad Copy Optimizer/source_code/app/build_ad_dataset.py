import os
import random
import math
from typing import List, Dict, Any

import requests
import pandas as pd
from dotenv import load_dotenv

# =========================
# CONFIG – EDIT THIS PART
# =========================

# Meta API
API_VERSION = "v18.0"
AD_ARCHIVE_URL = f"https://graph.facebook.com/{API_VERSION}/ads_archive"

# Country filter for Ad Library (ISO country codes)
AD_REACHED_COUNTRIES = ["US"]  # e.g. ["US", "CA"]

# Keywords to search in the Ad Library.
SEARCH_TERMS = [
    ("shoes", "ecommerce"),
    ("project management", "saas"),
    ("online course", "education"),
]

# Max ads per search term (Meta will cap; keep small at first)
ADS_PER_TERM = 50

# Synthetic CTR distributions by category (mean CTR in decimal)
CATEGORY_CTR_MEAN = {
    "ecommerce": 0.015,   # 1.5%
    "saas": 0.006,        # 0.6%
    "education": 0.01,    # 1.0%
}

# Synthetic CTR std dev
CATEGORY_CTR_STD = {
    "ecommerce": 0.005,
    "saas": 0.003,
    "education": 0.004,
}

# Impressions range
IMPRESSIONS_MIN = 5_000
IMPRESSIONS_MAX = 100_000

# CPC range (cost per click, in your currency)
CPC_MIN = 0.3
CPC_MAX = 2.0

# Output CSV path
OUTPUT_CSV = "meta_ads_dataset.csv"


# =========================
# UTILS
# =========================

def get_access_token() -> str:
    load_dotenv()
    token = os.getenv("META_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("META_ACCESS_TOKEN not found. Set it in .env.")
    return token


def meta_ad_library_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Simple wrapper around requests.get to call the Ad Library."""
    resp = requests.get(AD_ARCHIVE_URL, params=params)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Meta API error {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


# =========================
# STEP 1: FETCH ADS
# =========================

def fetch_ads_for_term(
    search_term: str,
    category: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch ads from Meta Ad Library for a given search term.
    NOTE: Ad Library has different requirements depending on the ad_type.
          For simplicity, we use 'ALL' and basic filters – adjust as needed.
    """
    access_token = get_access_token()

    fields = [
        "id",
        "ad_archive_id",
        "ad_creative_body",
        "ad_creative_link_title",
        "ad_creative_link_description",
        "page_name",
        "publisher_platforms",
        "ad_delivery_start_time",
        "ad_delivery_stop_time",
    ]

    params = {
        "access_token": access_token,
        "search_terms": search_term,
        "ad_reached_countries": ",".join(AD_REACHED_COUNTRIES),
        "ad_type": "ALL",
        "limit": min(limit, 100),  # Meta max page size 100
        "fields": ",".join(fields),
    }

    all_ads = []
    next_page_url = None

    while True:
        if next_page_url:
            resp = requests.get(next_page_url)
            if resp.status_code != 200:
                print("Error on pagination:", resp.text[:300])
                break
            data = resp.json()
        else:
            data = meta_ad_library_request(params)

        ads = data.get("data", [])
        for ad in ads:
            ad["_category_label"] = category
            ad["_search_term"] = search_term
        all_ads.extend(ads)

        if len(all_ads) >= limit:
            break

        paging = data.get("paging", {})
        next_page_url = paging.get("next")
        if not next_page_url:
            break

    return all_ads[:limit]


def fetch_all_ads() -> List[Dict[str, Any]]:
    """
    Fetch ads for all configured SEARCH_TERMS.
    Returns a list of raw Meta ad records.
    """
    all_ads = []
    for term, category in SEARCH_TERMS:
        print(f"Fetching ads for term='{term}', category='{category}'...")
        ads = fetch_ads_for_term(term, category, limit=ADS_PER_TERM)
        print(f"  -> got {len(ads)} ads")
        all_ads.extend(ads)
    print(f"Total raw ads fetched: {len(all_ads)}")
    return all_ads


# =========================
# STEP 2: SYNTHETIC CTR LABEL GENERATOR
# =========================

def sample_positive_truncated_normal(mean: float, std: float, low: float, high: float) -> float:
    """
    Sample from a normal distribution but truncated to [low, high].
    """
    for _ in range(10):
        x = random.gauss(mean, std)
        if low <= x <= high:
            return x
    # fallback if it keeps sampling outside
    return max(low, min(high, mean))


def generate_synthetic_metrics(category: str) -> Dict[str, Any]:
    """
    Generate synthetic impressions, clicks, conversions, and spend
    based on category-level CTR distributions.
    """
    impressions = random.randint(IMPRESSIONS_MIN, IMPRESSIONS_MAX)

    mean_ctr = CATEGORY_CTR_MEAN.get(category, 0.01)
    std_ctr = CATEGORY_CTR_STD.get(category, 0.004)
    ctr = sample_positive_truncated_normal(mean_ctr, std_ctr, 0.001, 0.08)  # 0.1% – 8%

    clicks = max(1, int(impressions * ctr))

    # Conversion rate as a fraction of clicks (2–25%)
    conv_rate = random.uniform(0.02, 0.25)
    conversions = max(0, int(clicks * conv_rate))

    # CPC between CPC_MIN and CPC_MAX
    cpc = random.uniform(CPC_MIN, CPC_MAX)
    spend = round(clicks * cpc, 2)

    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": spend,
    }


# =========================
# STEP 3: TRANSFORM TO YOUR CSV SCHEMA
# =========================

def to_schema_row(ad: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a raw Meta Ad Library record to your CSV schema, plus synthetic labels.
    """
    # Identity / grouping
    ad_id = ad.get("ad_archive_id") or ad.get("id")
    campaign_id = ""  # not exposed for public ads; keep empty

    # Textual content
    headline_text = (ad.get("ad_creative_link_title") or "").strip()
    body_text = (ad.get("ad_creative_body") or "").strip()

    # call_to_action not exposed in public library → leave blank
    call_to_action = ""

    # Context / metadata
    publisher_platforms = ad.get("publisher_platforms") or []
    platform = publisher_platforms[0].lower() if publisher_platforms else "facebook"

    placement = ""       # not available for public ads
    ad_format = "image"  # default guess; you can refine later

    category = ad.get("_category_label", "uncategorized")

    # Synthetic performance metrics
    metrics = generate_synthetic_metrics(category)

    row = {
        "ad_id": ad_id,
        "campaign_id": campaign_id,
        "headline_text": headline_text,
        "body_text": body_text,
        "call_to_action": call_to_action,
        "platform": platform,
        "placement": placement,
        "ad_format": ad_format,
        "category": category,
        "impressions": metrics["impressions"],
        "clicks": metrics["clicks"],
        "conversions": metrics["conversions"],
        "spend": metrics["spend"],
    }
    return row


def build_dataset(raw_ads: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw Meta ads to a pandas DataFrame matching your schema.
    Deduplicates by ad_id.
    """
    rows = [to_schema_row(ad) for ad in raw_ads]

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["ad_id"], inplace=True)

    # Optional: ensure column order
    columns = [
        "ad_id",
        "campaign_id",
        "headline_text",
        "body_text",
        "call_to_action",
        "platform",
        "placement",
        "ad_format",
        "category",
        "impressions",
        "clicks",
        "conversions",
        "spend",
    ]
    df = df[columns]

    return df


# =========================
# MAIN PIPELINE
# =========================

def main():
    raw_ads = fetch_all_ads()
    if not raw_ads:
        print("No ads fetched. Check your token / filters.")
        return

    df = build_dataset(raw_ads)
    print(f"Final dataset shape: {df.shape}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
