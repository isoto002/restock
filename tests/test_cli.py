"""The command line, tested the way a stranger will actually hit it.

Including the ways they will hit it wrong. A tool that answers a malformed CSV with a
stack trace tells its user nothing; these tests pin the clear-error behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from restock import __version__
from restock.cli import main


class TestDemo:
    def test_runs_end_to_end_and_prints_a_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`restock demo` is the one command in the README quickstart — it must just work."""
        exit_code = main(["demo"])
        out = capsys.readouterr().out

        assert exit_code == 0
        assert "REORDER PLAN" in out
        assert "WIDGET-BLUE" in out
        assert "Reorder point" in out or "REORDER POINT" in out.upper()

    def test_reports_forecast_accuracy(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A forecast with no accuracy figure beside it is a number nobody should trust."""
        main(["demo"])
        out = capsys.readouterr().out
        assert "MAPE" in out.upper()


class TestPlan:
    def test_reads_a_csv_and_prints_a_plan(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        csv = tmp_path / "sales.csv"
        rows = ["date,sku,units"]
        for day in range(1, 29):
            rows.append(f"2024-01-{day:02d},ACME-1,{10 + day % 7}")
        csv.write_text("\n".join(rows), encoding="utf-8")

        exit_code = main(["plan", "--input", str(csv), "--lead-time", "5"])
        out = capsys.readouterr().out

        assert exit_code == 0
        assert "ACME-1" in out

    def test_a_missing_file_is_a_clear_message_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["plan", "--input", str(tmp_path / "nope.csv")])
        err = capsys.readouterr().err

        assert exit_code == 2
        assert "not found" in err.lower()

    def test_missing_columns_name_what_is_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        csv = tmp_path / "wrong.csv"
        csv.write_text("day,product,qty\n2024-01-01,A,5\n", encoding="utf-8")

        exit_code = main(["plan", "--input", str(csv)])
        err = capsys.readouterr().err

        assert exit_code == 2
        assert "date" in err and "sku" in err and "units" in err

    def test_too_little_history_explains_itself(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Three days cannot support a weekly seasonal model — say so, do not guess."""
        csv = tmp_path / "short.csv"
        csv.write_text(
            "date,sku,units\n2024-01-01,A,5\n2024-01-02,A,6\n2024-01-03,A,7\n",
            encoding="utf-8",
        )

        exit_code = main(["plan", "--input", str(csv)])
        err = capsys.readouterr().err

        assert exit_code == 2
        assert "history" in err.lower() or "cycles" in err.lower()

    def test_rejects_an_impossible_service_level(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        csv = tmp_path / "sales.csv"
        csv.write_text("date,sku,units\n2024-01-01,A,5\n", encoding="utf-8")

        exit_code = main(["plan", "--input", str(csv), "--service-level", "1.0"])
        assert exit_code == 2
        assert "service" in capsys.readouterr().err.lower()


class TestVersion:
    def test_prints_the_package_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestNoCommand:
    def test_shows_help_rather_than_failing_silently(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main([])
        assert exit_code == 2
        assert "usage" in capsys.readouterr().err.lower()
