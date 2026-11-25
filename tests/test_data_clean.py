# tests/test_data_clean.py
# ─────────────────────────────────────────────────────────────
# Unit tests for the data cleaning pipeline.
#
# IMPORTANT: these tests do NOT need any CSV files.  They build
# tiny DataFrames inside each test so that pytest in CI (GitHub
# Actions) works without the gitignored data.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pytest

from mp028.data_clean import clean_train, clean_later, KEEP_COLS


# ── Helper: build a small fake training row ──────────────────
def _make_train_row(**overrides):
    """Return a dict for one training row with sensible defaults."""
    row = {
        "record_id": "R99901",
        "date": "2026-01-15",
        "ticket_price_rs": 250,
        "days_since_release": 10,
        "star_rating": 3.5,
        "show_hour": 14,
        "screen_seats": 200,
        "distance_to_metro_km": 2.5,
        "language": "hindi",
        "location": "Mall",
        "tickets_scanned_pct": 72.0,   # leaking column
        "occupancy_pct": 73.0,         # target
    }
    row.update(overrides)
    return row


def _make_later_row(**overrides):
    """Return a dict for one later.csv row."""
    row = {
        "record_id": "R29901",
        "date": "2026-07-10",
        "ticket_price_rs": 300,
        "days_since_release": 5,
        "star_rating": 4.0,
        "show_hour": 18,
        "screen_seats": 150,
        "distance_to_metro_km": 3.0,
        "language": "english",
        "location": "Highway",
        "tickets_scanned_pct": 65.0,
        "occupancy_pct": 66.0,
    }
    row.update(overrides)
    return row


# ═════════════════════════════════════════════════════════════
# Tests for clean_train
# ═════════════════════════════════════════════════════════════

class TestCleanTrain:
    """Tests for the clean_train function."""

    def test_leaking_column_removed(self):
        """tickets_scanned_pct must be dropped from the output."""
        df = pd.DataFrame([_make_train_row()])
        result = clean_train(df)
        assert "tickets_scanned_pct" not in result.columns

    def test_output_columns(self):
        """Output must have exactly the expected columns."""
        df = pd.DataFrame([_make_train_row()])
        result = clean_train(df)
        assert list(result.columns) == KEEP_COLS

    def test_duplicates_removed(self):
        """Fully duplicated rows should be dropped."""
        row = _make_train_row()
        df = pd.DataFrame([row, row, row])          # 3 copies → keep 1
        result = clean_train(df)
        assert len(result) == 1

    def test_sentinel_replaced(self):
        """ticket_price_rs == 999 should be replaced with NaN (no imputation)."""
        rows = [
            _make_train_row(record_id="R001", ticket_price_rs=200),
            _make_train_row(record_id="R002", ticket_price_rs=300),
            _make_train_row(record_id="R003", ticket_price_rs=999),
        ]
        df = pd.DataFrame(rows)
        result = clean_train(df)
        # 999 must be gone and replaced with NaN
        assert 999 not in result["ticket_price_rs"].values
        assert pd.isna(
            result.loc[result["record_id"] == "R003", "ticket_price_rs"].iloc[0]
        )

    def test_target_typos_fixed(self):
        """Targets > 200 should be divided by 10."""
        rows = [
            _make_train_row(record_id="R001", occupancy_pct=73.0),
            _make_train_row(record_id="R002", occupancy_pct=730.0),
        ]
        df = pd.DataFrame(rows)
        result = clean_train(df)
        fixed_val = result.loc[
            result["record_id"] == "R002", "occupancy_pct"
        ].iloc[0]
        assert fixed_val == 73.0

    def test_sorted_by_date(self):
        """Output must be sorted by date (ascending)."""
        rows = [
            _make_train_row(record_id="R001", date="2026-03-01"),
            _make_train_row(record_id="R002", date="2026-01-15"),
        ]
        df = pd.DataFrame(rows)
        result = clean_train(df)
        assert list(result["record_id"]) == ["R002", "R001"]

    def test_date_is_datetime(self):
        """The date column must be datetime after cleaning."""
        df = pd.DataFrame([_make_train_row()])
        result = clean_train(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_text_normalisation(self):
        """Stray whitespace, casing, and punctuation in language/location are normalised."""
        rows = [
            _make_train_row(record_id="R001", language=" Hindi! ", location="mall "),
            _make_train_row(record_id="R002", language="ENGLISH.", location="city centre."),
        ]
        df = pd.DataFrame(rows)
        result = clean_train(df)
        assert list(result["language"]) == ["hindi", "english"]
        assert list(result["location"]) == ["Mall", "City Centre"]

    def test_unit_mixup_fixed(self):
        """A location group with a 1000x unit mixup is divided by 1000."""
        rows = [
            _make_train_row(record_id="R001", location="City Centre", distance_to_metro_km=2.0),
            _make_train_row(record_id="R002", location="Highway", distance_to_metro_km=2.0),
            _make_train_row(record_id="R003", location="Mall", distance_to_metro_km=2000.0),
        ]
        df = pd.DataFrame(rows)
        result = clean_train(df)
        mall_val = result.loc[result["record_id"] == "R003", "distance_to_metro_km"].iloc[0]
        assert mall_val == 2.0

    def test_stuck_sensor_masked(self):
        """A stuck sensor run of >= 10 consecutive dates is set to NaN."""
        rows = [
            _make_train_row(record_id=f"R{i:03d}", date=f"2026-02-{i:02d}",
                            ticket_price_rs=250.0 if 3 <= i <= 12 else 150.0 + i * 10)
            for i in range(1, 15)
        ]
        df = pd.DataFrame(rows)
        result = clean_train(df)
        stuck_mask = (result["date"] >= "2026-02-03") & (result["date"] <= "2026-02-12")
        assert result.loc[stuck_mask, "ticket_price_rs"].isna().all()
        assert result.loc[~stuck_mask, "ticket_price_rs"].notna().all()


# ═════════════════════════════════════════════════════════════
# Tests for clean_later
# ═════════════════════════════════════════════════════════════

class TestCleanLater:
    """Tests for the clean_later function."""

    def test_leaking_column_removed(self):
        """tickets_scanned_pct must be dropped."""
        df = pd.DataFrame([_make_later_row()])
        result = clean_later(df)
        assert "tickets_scanned_pct" not in result.columns

    def test_output_columns(self):
        """Output must have exactly the expected columns."""
        df = pd.DataFrame([_make_later_row()])
        result = clean_later(df)
        assert list(result.columns) == KEEP_COLS

    def test_sorted_by_date(self):
        """Output must be sorted by date."""
        rows = [
            _make_later_row(record_id="R001", date="2026-08-01"),
            _make_later_row(record_id="R002", date="2026-07-01"),
        ]
        df = pd.DataFrame(rows)
        result = clean_later(df)
        assert list(result["record_id"]) == ["R002", "R001"]
