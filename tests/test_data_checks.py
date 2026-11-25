# tests/test_data_checks.py
# ─────────────────────────────────────────────────────────────
# Unit tests for all seven detectors in mp028.data_checks.
#
# Uses tiny hand-made DataFrames built inside each test so that
# pytest runs without any CSV files from data/raw/.
# ─────────────────────────────────────────────────────────────

import pandas as pd

from mp028.data_checks import (
    check_leaking_column,
    check_unit_mixup,
    check_sentinel_value,
    check_duplicate_rows,
    check_stuck_sensor,
    check_target_typos,
    check_spelling_variants,
)


def test_leaking_column():
    """Detect a post-event column missing from holdout; clean case returns False."""
    train_df = pd.DataFrame({
        "ticket_price_rs": [200, 300, 400],
        "tickets_scanned_pct": [60.0, 75.0, 90.0],
        "occupancy_pct": [60.1, 75.2, 89.9],
    })
    holdout_df = pd.DataFrame({
        "ticket_price_rs": [250, 350],
    })

    res = check_leaking_column(train_df, holdout_df)
    assert set(res.keys()) == {"present", "column"}
    assert res["present"] is True
    assert res["column"] == "tickets_scanned_pct"

    # Negative case: no extra column beyond target
    clean_train_df = train_df.drop(columns=["tickets_scanned_pct"])
    res_clean = check_leaking_column(clean_train_df, holdout_df)
    assert set(res_clean.keys()) == {"present", "column"}
    assert res_clean["present"] is False
    assert res_clean["column"] is None


def test_sentinel_value():
    """Detect repeated 999 sentinel values (>= 5 rows); clean column returns False."""
    df_bad = pd.DataFrame({
        "ticket_price_rs": [200, 250, 999, 999, 999, 999, 999, 999],
    })
    res = check_sentinel_value(df_bad)
    assert set(res.keys()) == {"present", "column", "value", "count"}
    assert res["present"] is True
    assert res["column"] == "ticket_price_rs"
    assert res["value"] == 999
    assert res["count"] == 6

    df_clean = pd.DataFrame({
        "ticket_price_rs": [200, 250, 300, 350, 400, 450],
    })
    res_clean = check_sentinel_value(df_clean)
    assert set(res_clean.keys()) == {"present", "column", "value", "count"}
    assert res_clean["present"] is False
    assert res_clean["count"] is None


def test_duplicate_rows():
    """Count 2 extra duplicate copies; unique rows return False."""
    df_dups = pd.DataFrame({
        "record_id": ["R1", "R1", "R1", "R2"],
        "ticket_price_rs": [200, 200, 200, 300],
    })
    res = check_duplicate_rows(df_dups)
    assert set(res.keys()) == {"present", "count"}
    assert res["present"] is True
    assert res["count"] == 2

    df_clean = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "ticket_price_rs": [200, 300, 400],
    })
    res_clean = check_duplicate_rows(df_clean)
    assert set(res_clean.keys()) == {"present", "count"}
    assert res_clean["present"] is False
    assert res_clean["count"] is None


def test_target_typos():
    """Detect targets typed 10x too big (> 200); normal targets return False."""
    df_typos = pd.DataFrame({
        "occupancy_pct": [25.0, 55.0, 80.0, 500.0, 700.0],
    })
    res = check_target_typos(df_typos)
    assert set(res.keys()) == {"present", "count"}
    assert res["present"] is True
    assert res["count"] == 2

    df_clean = pd.DataFrame({
        "occupancy_pct": [20.0, 45.0, 85.0, 100.0],
    })
    res_clean = check_target_typos(df_clean)
    assert set(res_clean.keys()) == {"present", "count"}
    assert res_clean["present"] is False
    assert res_clean["count"] is None


def test_unit_mixup():
    """Detect a location group recorded in a 1000x larger unit; similar groups return False."""
    df_mixup = pd.DataFrame({
        "location": ["City Centre", "City Centre", "Highway", "Highway", "Mall", "Mall"],
        "distance_to_metro_km": [1.5, 2.5, 1.8, 2.2, 1800.0, 2200.0],
    })
    res = check_unit_mixup(df_mixup)
    assert set(res.keys()) == {"present", "column", "group_value", "factor"}
    assert res["present"] is True
    assert res["column"] == "distance_to_metro_km"
    assert res["group_value"] == "Mall"
    assert res["factor"] == 1000

    df_clean = pd.DataFrame({
        "location": ["City Centre", "City Centre", "Highway", "Highway", "Mall", "Mall"],
        "distance_to_metro_km": [1.5, 2.5, 1.8, 2.2, 2.0, 3.0],
    })
    res_clean = check_unit_mixup(df_clean)
    assert set(res_clean.keys()) == {"present", "column", "group_value", "factor"}
    assert res_clean["present"] is False
    assert res_clean["factor"] is None


def test_stuck_sensor():
    """Detect a long consecutive run of identical values across dates; varied values return False."""
    dates = [f"2026-02-{day:02d}" for day in range(1, 16)]
    # 10 consecutive days stuck at 250.0 from 2026-02-03 to 2026-02-12
    prices = [180.0, 210.0] + [250.0] * 10 + [290.0, 310.0, 330.0]
    df_stuck = pd.DataFrame({
        "date": dates,
        "ticket_price_rs": prices,
    })
    res = check_stuck_sensor(df_stuck)
    assert set(res.keys()) == {"present", "column", "start_date", "end_date"}
    assert res["present"] is True
    assert res["column"] == "ticket_price_rs"
    assert res["start_date"] == "2026-02-03"
    assert res["end_date"] == "2026-02-12"

    df_varied = pd.DataFrame({
        "date": dates,
        "ticket_price_rs": [150.0 + i * 10 for i in range(len(dates))],
    })
    res_clean = check_stuck_sensor(df_varied)
    assert set(res_clean.keys()) == {"present", "column", "start_date", "end_date"}
    assert res_clean["present"] is False
    assert res_clean["start_date"] is None
    assert res_clean["end_date"] is None


def test_spelling_variants():
    """Detect spelling variants in a text column compared to clean later_df."""
    later_df = pd.DataFrame({
        "location": ["Mall", "Highway", "City Centre"],
    })
    train_bad = pd.DataFrame({
        "location": ["Mall", "mall ", "MALL.", "Highway", "City Centre"],
    })
    res = check_spelling_variants(train_bad, later_df)
    assert set(res.keys()) == {"present", "column", "true_levels"}
    assert res["present"] is True
    assert res["column"] == "location"
    assert res["true_levels"] == 3

    train_clean = pd.DataFrame({
        "location": ["Mall", "Highway", "City Centre", "Mall"],
    })
    res_clean = check_spelling_variants(train_clean, later_df)
    assert set(res_clean.keys()) == {"present", "column", "true_levels"}
    assert res_clean["present"] is False
    assert res_clean["true_levels"] is None


def test_main_output_structure(monkeypatch, capsys):
    """Verify main() prints all 7 check sections and writes findings without reading CSVs."""
    import mp028.data_checks as dc

    train_df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "ticket_price_rs": [200, 300, 400],
        "days_since_release": [5, 10, 15],
        "star_rating": [3.5, 4.0, 4.5],
        "show_hour": [12, 15, 18],
        "screen_seats": [150, 200, 250],
        "distance_to_metro_km": [1.5, 2.5, 3.5],
        "language": ["hindi", "english", "regional"],
        "location": ["Mall", "Highway", "City Centre"],
        "tickets_scanned_pct": [60.0, 70.0, 80.0],
        "occupancy_pct": [60.0, 70.0, 80.0],
    })
    later_df = train_df.copy()
    holdout_df = train_df.drop(columns=["tickets_scanned_pct", "occupancy_pct"])

    frames = iter([train_df, later_df, holdout_df])
    monkeypatch.setattr(dc.pd, "read_csv", lambda *_a, **_kw: next(frames))

    recorded = {}
    monkeypatch.setattr(dc, "update_findings", lambda block, val: recorded.update({block: val}))

    results = dc.main()
    out = capsys.readouterr().out

    for i, kind in enumerate([
        "leaking_column", "unit_mixup", "sentinel_value",
        "duplicate_rows", "stuck_sensor", "target_typos", "spelling_variants",
    ], start=1):
        assert f"=== {i}. {kind} ===" in out
        assert f"data_issues.{kind}" in recorded
        assert kind in results
