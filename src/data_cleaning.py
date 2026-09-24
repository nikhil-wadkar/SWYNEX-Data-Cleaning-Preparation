"""
SWYNEX Task 1 - Data Cleaning & Preparation
Dataset: Customer Personality Analysis / Marketing Campaign

Run:
    python src/data_cleaning.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "marketing_campaign.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "customer_personality_cleaned.csv"

df = pd.read_csv(RAW_PATH, sep=None, engine="python")

# 1. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# 2. Remove exact duplicate rows and duplicate customer IDs
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["id"], keep="first")

# 3. Convert customer date to datetime
df["dt_customer"] = pd.to_datetime(
    df["dt_customer"], dayfirst=True, errors="coerce"
)

# 4. Standardize categorical values
df["education"] = df["education"].astype("string").str.strip().replace({
    "2n Cycle": "2nd Cycle"
})
df["marital_status"] = df["marital_status"].astype("string").str.strip().replace({
    "Alone": "Other",
    "Absurd": "Other",
    "YOLO": "Other"
})

# 5. Handle implausible birth years
df.loc[df["year_birth"] < 1900, "year_birth"] = np.nan
df["year_birth"] = (
    df["year_birth"].fillna(df["year_birth"].median()).round().astype("Int64")
)

# 6. Handle missing income using the median
df["income"] = df["income"].fillna(df["income"].median()).round(2)

# 7. Convert count/indicator fields to integer type
integer_cols = [
    "id", "kidhome", "teenhome", "recency",
    "mnt_wines", "mnt_fruits", "mnt_meat_products", "mnt_fish_products",
    "mnt_sweet_products", "mnt_gold_prods", "num_deals_purchases",
    "num_web_purchases", "num_catalog_purchases", "num_store_purchases",
    "num_web_visits_month", "accepted_cmp3", "accepted_cmp4", "accepted_cmp5",
    "accepted_cmp1", "accepted_cmp2", "complain", "z_cost_contact",
    "z_revenue", "response"
]
for col in integer_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# 8. Export cleaned dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values remaining: {int(df.isna().sum().sum())}")
print(f"Duplicate rows remaining: {int(df.duplicated().sum())}")
