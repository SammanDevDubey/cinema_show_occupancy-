# src/mp028/data_checks.py
# ─────────────────────────────────────────────────────────────
# Functions that detect each of the SEVEN data problems listed
# in the Student Handbook (section 7).  Every finding is
# computed from the data, never hard-coded.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

from mp028.config import load_params, REPO_ROOT
from mp028.findings import update_findings


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


def _longest_run(series: pd.Series) -> int:
    """Return length of the longest consecutive identical-value run."""
    vals = series.values
    if len(vals) == 0:
        return 0
    best = cur = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def main() -> dict:
    """Read raw CSVs, print evidence for all 7 checks, and update findings.yaml."""
    params = load_params()
    train_df = pd.read_csv(REPO_ROOT / params["paths"]["train_raw"])
    later_df = pd.read_csv(REPO_ROOT / params["paths"]["later_raw"])
    holdout_df = pd.read_csv(REPO_ROOT / params["paths"]["holdout"])
    target = params["data"]["target"]

    results = run_all_checks(train_df, later_df, holdout_df)

    # 1. Leaking column evidence
    extra_cols = sorted(set(train_df.columns) - set(holdout_df.columns) - {target})
    print("=== 1. leaking_column ===")
    print(f"Missing from holdout (excl. target): {extra_cols}")
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            corr = train_df[col].corr(train_df[target])
            print(f"  corr({col}, {target}) = {corr:.4f}")
            no_typos = train_df.drop_duplicates()
            no_typos = no_typos[no_typos[target] <= 100]
            corr_clean = no_typos[col].corr(no_typos[target])
            print(f"  corr (typos and duplicates excluded) = {corr_clean:.4f}")
    print("Result:", results["leaking_column"])

    # 2. Unit mixup evidence
    print("\n=== 2. unit_mixup ===")
    num_cols = [c for c in train_df.select_dtypes(include="number").columns if c != target]
    for col in num_cols:
        stats = train_df.groupby("location")[col].agg(["min", "mean", "max"]).round(2)
        print(f"  {col} by location:\n{stats.to_string()}")
    print("Result:", results["unit_mixup"])

    # 3. Sentinel value evidence
    print("\n=== 3. sentinel_value ===")
    for col in num_cols:
        top3 = train_df[col].value_counts().head(3).to_dict()
        print(f"  {col} top repeated: {top3}")
    print("Result:", results["sentinel_value"])

    # 4. Duplicate rows evidence
    print("\n=== 4. duplicate_rows ===")
    print(f"  df.duplicated().sum() = {int(train_df.duplicated().sum())}")
    print("Result:", results["duplicate_rows"])

    # 5. Stuck sensor evidence
    print("\n=== 5. stuck_sensor ===")
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    for col in num_cols:
        print(f"  {col} longest identical run: {_longest_run(sorted_df[col])}")
    print("Result:", results["stuck_sensor"])

    # 6. Target typos evidence
    print("\n=== 6. target_typos ===")
    top_targets = sorted(train_df[target].tolist(), reverse=True)[:12]
    print(f"  Top 12 {target} values: {top_targets}")
    if len(top_targets) > 1:
        diffs = [round(top_targets[i] - top_targets[i + 1], 2) for i in range(len(top_targets) - 1)]
        print(f"  Consecutive gaps: {diffs}")
    print("Result:", results["target_typos"])

    # 7. Spelling variants evidence
    print("\n=== 7. spelling_variants ===")
    for col in ["language", "location"]:
        print(f"  {col} value_counts: {train_df[col].value_counts().to_dict()}")
    print("Result:", results["spelling_variants"])

    for kind, res in results.items():
        update_findings(f"data_issues.{kind}", res)

    return results


if __name__ == "__main__":
    main()
