"""The upload parser, which is the first thing a stranger's own data touches.

Every test here is a file somebody will actually hand it: the wrong columns, a transaction
export with many rows per day, text where numbers should be, and too little history. Each
has to produce a sentence a non-engineer can act on rather than a traceback.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# app.py sits at the repo root rather than in the package, so load it by path.
_SPEC = importlib.util.spec_from_file_location(
    "restock_app", Path(__file__).parent.parent / "app.py"
)
assert _SPEC and _SPEC.loader
app = importlib.util.module_from_spec(_SPEC)
sys.modules["restock_app"] = app
_SPEC.loader.exec_module(app)


def csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def daily_rows(sku: str, days: int, start: str = "2025-01-01") -> str:
    dates = pd.date_range(start, periods=days, freq="D")
    return "".join(f"{d.date()},{sku},{10 + i % 5}\n" for i, d in enumerate(dates))


HEADER = "date,sku,units\n"


class TestItAccepts:
    def test_a_clean_file(self) -> None:
        frame = app.read_uploaded_history(csv_bytes(HEADER + daily_rows("WIDGET", 60)))
        assert list(frame.columns) == ["sku", "date", "units"]
        assert len(frame) == 60

    def test_headers_in_any_case_with_padding(self) -> None:
        """Real exports arrive as ' Date ', 'SKU', 'Units'."""
        body = daily_rows("WIDGET", 60)
        frame = app.read_uploaded_history(csv_bytes(" Date ,SKU,UNITS\n" + body))
        assert len(frame) == 60

    def test_extra_columns_are_ignored(self) -> None:
        """Nobody's export has exactly three columns."""
        rows = "".join(f"2025-01-{d:02d},WIDGET,12,store-1,99.5\n" for d in range(1, 29))
        rows += "".join(f"2025-02-{d:02d},WIDGET,12,store-1,99.5\n" for d in range(1, 29))
        frame = app.read_uploaded_history(csv_bytes("date,sku,units,store,price\n" + rows))
        assert list(frame.columns) == ["sku", "date", "units"]

    def test_many_rows_per_day_are_summed_not_dropped(self) -> None:
        """A transaction-level export has one row per ORDER, not per day.

        Taking the last row would silently under-report demand, which is the kind of quiet
        wrongness this whole project exists to avoid.
        """
        dates = pd.date_range("2025-01-01", periods=50, freq="D")
        rows = "".join(f"{d.date()},WIDGET,5\n{d.date()},WIDGET,7\n" for d in dates)
        frame = app.read_uploaded_history(csv_bytes(HEADER + rows))
        assert len(frame) == 50, "one row per day after aggregation"
        assert (frame["units"] == 12).all(), "5 + 7, not 7"

    def test_short_skus_are_dropped_but_long_ones_survive(self) -> None:
        body = daily_rows("GOOD", 60) + "2025-01-01,TOOSHORT,5\n2025-01-02,TOOSHORT,6\n"
        frame = app.read_uploaded_history(csv_bytes(HEADER + body))
        assert set(frame["sku"]) == {"GOOD"}


class TestItRefusesWithAReason:
    def test_a_missing_column_names_it(self) -> None:
        with pytest.raises(app.UploadError, match="units"):
            app.read_uploaded_history(csv_bytes("date,sku\n2025-01-01,WIDGET\n"))

    def test_a_file_that_is_not_csv(self) -> None:
        with pytest.raises(app.UploadError):
            app.read_uploaded_history(b"\x00\x01\x02 not a csv at all")

    def test_text_where_numbers_belong(self) -> None:
        rows = "".join(f"2025-01-{d:02d},WIDGET,lots\n" for d in range(1, 29))
        with pytest.raises(app.UploadError, match="No usable rows"):
            app.read_uploaded_history(csv_bytes(HEADER + rows))

    def test_too_little_history_says_how_much_is_needed(self) -> None:
        with pytest.raises(app.UploadError, match="fewer than"):
            app.read_uploaded_history(csv_bytes(HEADER + daily_rows("WIDGET", 5)))

    def test_the_error_is_a_sentence_not_a_traceback(self) -> None:
        """What a person sees matters as much as that it failed."""
        with pytest.raises(app.UploadError) as caught:
            app.read_uploaded_history(csv_bytes("wrong,headers\n1,2\n"))
        message = str(caught.value)
        assert message.endswith(".") and "date, sku, units" in message


class TestTheCommittedSampleStillWorks:
    def test_the_repos_own_sample_csv_parses(self) -> None:
        """The file the README tells people to copy. If this breaks, the docs lie."""
        sample = Path(__file__).parent.parent / "data" / "sample_sales.csv"
        frame = app.read_uploaded_history(sample.read_bytes())
        assert not frame.empty
        assert frame["sku"].nunique() >= 1
