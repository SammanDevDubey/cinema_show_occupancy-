# src/mp028/data_checks.py
# ─────────────────────────────────────────────────────────────
# Functions that detect each of the SEVEN data problems listed
# in the Student Handbook (section 7).  Every finding is
# computed from the data, never hard-coded.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np


# ── 1. Leaking column ───────────────────────────────────────
def check_leaking_column(train_df: pd.DataFrame,
                         holdout_df: pd.DataFrame,
                         target: str = "occupancy_pct") -> dict:
    """Find columns in train that are missing from holdout (besides target).

    A leaking column is one that is only filled in AFTER the event,
    so it secretly contains the answer.  It will be absent from the
    holdout file because the event has not happened yet.

    Returns
    -------
    dict  with keys ``present`` (bool) and ``column`` (str or None).
    """
    # Columns in train but not in holdout, excluding the target itself
    extra = set(train_df.columns) - set(holdout_df.columns) - {target}
    if extra:
        # Pick the one column (there should be exactly one)
        col = sorted(extra)[0]
        return {"present": True, "column": col}
    return {"present": False, "column": None}


# ── 2. Unit mixup ───────────────────────────────────────────
def check_unit_mixup(df: pd.DataFrame,
                     group_col: str = "location",
                     check_cols: list[str] | None = None,
                     threshold: float = 5.0) -> dict:
    """Check if one group recorded a numeric column in a different unit.

    Compares the mean of each numeric column across groups.  If one
    group's mean is *threshold* times larger or smaller than the
    other groups, that is a mixup.

    Returns
    -------
    dict  with ``present``, ``column``, ``group_value``, ``factor``.
    """
    if check_cols is None:
        check_cols = df.select_dtypes(include="number").columns.tolist()

    for col in check_cols:
        grouped_mean = df.groupby(group_col)[col].mean()
        overall_median = grouped_mean.median()
        if overall_median == 0:
            continue
        for group_val, group_mean in grouped_mean.items():
            ratio = group_mean / overall_median
            if ratio > threshold:
                # Group is *factor* times too big
                factor = round(ratio)
                return {"present": True, "column": col,
                        "group_value": group_val, "factor": factor}
            if ratio < 1 / threshold:
                factor = round(1 / ratio)
                return {"present": True, "column": col,
                        "group_value": group_val, "factor": factor}

    return {"present": False, "column": None,
            "group_value": None, "factor": None}


# ── 3. Sentinel value ───────────────────────────────────────
def check_sentinel_value(df: pd.DataFrame,
                         suspects: list[int] | None = None) -> dict:
    """Look for special numbers used instead of blanks.

    Common sentinel values are -1, -999, 999, 9999.

    Returns
    -------
    dict  with ``present``, ``column``, ``value``, ``count``.
    """
    if suspects is None:
        suspects = [-1, -999, 999, 9999]

    for col in df.select_dtypes(include="number").columns:
        vc = df[col].value_counts()
        for s in suspects:
            if s in vc.index and vc[s] >= 5:  # at least 5 occurrences
                return {"present": True, "column": col,
                        "value": int(s), "count": int(vc[s])}

    return {"present": False, "column": None,
            "value": None, "count": None}


# ── 4. Duplicate rows ───────────────────────────────────────
def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Count fully duplicated rows (extra copies you would delete).

    Returns
    -------
    dict  with ``present`` and ``count``.
    """
    n_dups = int(df.duplicated().sum())
    return {"present": n_dups > 0, "count": n_dups if n_dups > 0 else None}


# ── 5. Stuck sensor ─────────────────────────────────────────
def check_stuck_sensor(df: pd.DataFrame,
                       date_col: str = "date",
                       min_run: int = 10) -> dict:
    """Find a column with the same value repeated for many consecutive days.

    The DataFrame is sorted by date first.

    Returns
    -------
    dict  with ``present``, ``column``, ``start_date``, ``end_date``.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    num_cols = df_sorted.select_dtypes(include="number").columns

    for col in num_cols:
        vals = df_sorted[col].values
        dates = df_sorted[date_col].values
        best_run = 1
        best_start = 0
        cur_run = 1
        cur_start = 0
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1]:
                cur_run += 1
            else:
                if cur_run > best_run:
                    best_run = cur_run
                    best_start = cur_start
                cur_run = 1
                cur_start = i
        # check last run
        if cur_run > best_run:
            best_run = cur_run
            best_start = cur_start

        if best_run >= min_run:
            return {
                "present": True,
                "column": col,
                "start_date": str(dates[best_start]),
                "end_date": str(dates[best_start + best_run - 1]),
            }

    return {"present": False, "column": None,
            "start_date": None, "end_date": None}


# ── 6. Target typos ─────────────────────────────────────────
def check_target_typos(df: pd.DataFrame,
                       target: str = "occupancy_pct",
                       cutoff: float = 200.0) -> dict:
    """Find rows where the target was typed with an extra zero (10x too big).

    Any target value above *cutoff* is assumed to be a typo because
    occupancy_pct is a percentage (0-100).

    Returns
    -------
    dict  with ``present`` and ``count``.
    """
    bad = df[df[target] > cutoff]
    n = len(bad)
    return {"present": n > 0, "count": n if n > 0 else None}


# ── 7. Spelling variants ────────────────────────────────────
def check_spelling_variants(df: pd.DataFrame,
                            later_df: pd.DataFrame,
                            text_cols: list[str] | None = None) -> dict:
    """Check if text columns have misspelled categories.

    Uses later.csv (which is clean) as the reference for the correct
    category names.

    Returns
    -------
    dict  with ``present``, ``column``, ``true_levels``.
    """
    if text_cols is None:
        # Only check actual categorical columns, not IDs or dates
        text_cols = [
            c for c in df.select_dtypes(include="object").columns
            if c not in ("record_id", "date")
        ]

    for col in text_cols:
        # Strip whitespace and lowercase for comparison
        train_levels = set(df[col].str.strip().str.lower().unique())
        later_levels = set(later_df[col].str.strip().str.lower().unique())
        # If train has MORE unique values after normalising, there are variants
        if len(train_levels) > len(later_levels):
            return {
                "present": True,
                "column": col,
                "true_levels": len(later_levels),
            }

    return {"present": False, "column": None, "true_levels": None}


# ── Run all checks ──────────────────────────────────────────
def run_all_checks(train_df: pd.DataFrame,
                   later_df: pd.DataFrame,
                   holdout_df: pd.DataFrame) -> dict:
    """Run all seven data-quality checks and return a combined dict.

    Returns
    -------
    dict  keyed by check name, each containing the check's result.
    """
    return {
        "leaking_column":    check_leaking_column(train_df, holdout_df),
        "unit_mixup":        check_unit_mixup(train_df),
        "sentinel_value":    check_sentinel_value(train_df),
        "duplicate_rows":    check_duplicate_rows(train_df),
        "stuck_sensor":      check_stuck_sensor(train_df),
        "target_typos":      check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df, later_df),
    }
