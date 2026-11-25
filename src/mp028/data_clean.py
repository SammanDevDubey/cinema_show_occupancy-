# src/mp028/data_clean.py
# ─────────────────────────────────────────────────────────────
# Cleaning functions for train.csv and later.csv.
#
# clean_train(df) handles all the messy issues found by
# data_checks.py:  removes the leaking column, fixes sentinel
# values, drops duplicates, corrects target typos.
#
# clean_later(df) is simpler because later.csv is already clean
# (just drops the leaking column).
#
# Both return a DataFrame with columns:
#   record_id, date (datetime), the 8 FEATURES, occupancy_pct
# sorted by date.
# ─────────────────────────────────────────────────────────────

import re
from pathlib import Path

import pandas as pd
import numpy as np

from mp028.config import load_params, REPO_ROOT
from mp028.data_checks import check_unit_mixup, check_stuck_sensor


# The column that leaks the answer (filled in after the event)
LEAKING_COL = "tickets_scanned_pct"

# Features the model is allowed to use
FEATURES = [
    "ticket_price_rs", "days_since_release", "star_rating",
    "show_hour", "screen_seats", "distance_to_metro_km",
    "language", "location",
]

TARGET = "occupancy_pct"

# Columns to keep in the cleaned output
KEEP_COLS = ["record_id", "date"] + FEATURES + [TARGET]


_LANG_MAP = {
    "hindi": "hindi",
    "regional": "regional",
    "english": "english",
}
_LOC_MAP = {
    "city centre": "City Centre",
    "mall": "Mall",
    "highway": "Highway",
}


def _normalise_text(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Strip, lowercase, remove stray punctuation, and map to canonical labels."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda s: re.sub(r"[^a-z\s]", "", s).strip())
        .apply(lambda s: re.sub(r"\s+", " ", s))
    )
    return cleaned.map(lambda s: mapping.get(s, s))


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw training data (train.csv).

    Steps (in order):
      1. Drop the leaking column (tickets_scanned_pct).
      2. Drop fully duplicated rows, keeping first occurrence.
      3. Replace the sentinel value 999 in ticket_price_rs with NaN
         (imputation is left to the modelling pipeline).
      4. Fix target typos: values > 200 are divided by 10 (extra zero).
      5. Convert the date column to datetime.
      6. Keep only the required columns, sort by date.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame read from train.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for modelling.
    """
    out = df.copy()

    # 1. Remove the leaking column
    if LEAKING_COL in out.columns:
        out = out.drop(columns=[LEAKING_COL])

    # 2. Remove duplicate rows
    out = out.drop_duplicates()

    # 3. Replace sentinel value 999 in ticket_price_rs with NaN
    out["ticket_price_rs"] = out["ticket_price_rs"].replace(999, np.nan)

    # 4. Fix target typos (values typed with an extra zero)
    out.loc[out[TARGET] > 200, TARGET] = (
        out.loc[out[TARGET] > 200, TARGET] / 10
    )

    # 5. Normalise language and location categories
    out["language"] = _normalise_text(out["language"], _LANG_MAP)
    out["location"] = _normalise_text(out["location"], _LOC_MAP)

    # 6. Fix unit mixup if detected
    mixup = check_unit_mixup(out)
    if mixup["present"]:
        mask = out["location"] == mixup["group_value"]
        out.loc[mask, mixup["column"]] = (
            out.loc[mask, mixup["column"]] / mixup["factor"]
        )

    # 7. Mask stuck sensor runs as NaN if detected
    stuck = check_stuck_sensor(out)
    if stuck["present"]:
        dates_dt = pd.to_datetime(out["date"])
        mask = (
            (dates_dt >= pd.to_datetime(stuck["start_date"]))
            & (dates_dt <= pd.to_datetime(stuck["end_date"]))
        )
        out.loc[mask, stuck["column"]] = np.nan

    # 8. Parse dates
    out["date"] = pd.to_datetime(out["date"])

    # 9. Keep only required columns, sort by date
    out = out[KEEP_COLS].sort_values("date").reset_index(drop=True)

    return out


def clean_later(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the later data (later.csv).

    later.csv is already clean, so we only need to:
      1. Drop the leaking column.
      2. Convert the date column to datetime.
      3. Keep only the required columns, sort by date.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame read from later.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for modelling / monitoring.
    """
    out = df.copy()

    # 1. Remove the leaking column
    if LEAKING_COL in out.columns:
        out = out.drop(columns=[LEAKING_COL])

    # 2. Parse dates
    out["date"] = pd.to_datetime(out["date"])

    # 3. Keep only required columns, sort by date
    out = out[KEEP_COLS].sort_values("date").reset_index(drop=True)

    return out


def save_cleaned() -> Path:
    """Read raw train.csv, clean it, save to data/processed/train_clean.csv, and print a summary."""
    params = load_params()
    raw_path = REPO_ROOT / params["paths"]["train_raw"]
    raw_df = pd.read_csv(raw_path)

    rows_before = len(raw_df)
    dups_removed = int(raw_df.duplicated().sum())
    dedup_df = raw_df.drop_duplicates()
    sentinels_to_nan = int((dedup_df["ticket_price_rs"] == 999).sum())
    typos_fixed = int((dedup_df[TARGET] > 200).sum())

    clean_df = clean_train(raw_df)

    out_path = REPO_ROOT / "data" / "processed" / "train_clean.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)

    print(f"Rows before: {rows_before}, Rows after: {len(clean_df)}")
    print(f"Duplicates removed: {dups_removed}")
    print(f"Sentinel values set to NaN: {sentinels_to_nan}")
    print(f"Typo targets divided by 10: {typos_fixed}")
    print("NaN count per column:")
    print(clean_df.isna().sum().to_string())
    return out_path


if __name__ == "__main__":
    save_cleaned()
