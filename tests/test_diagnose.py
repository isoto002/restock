"""The input check, written against the data that exposed the need for it.

The headline case is `test_the_stock_level_series_that_fooled_it`: a real inventory sheet
loaded as demand, which the tool happily planned against. That series has to be refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from restock.diagnose import (
    MIN_OBSERVATIONS,
    Shape,
    diagnose,
)
from restock.simulate import generate_sales


def smooth(days: int = 200, seed: int = 3) -> np.ndarray:
    """A normal product: sells most days, some weekly shape, some noise."""
    rng = np.random.default_rng(seed)
    return np.maximum(0, 40 + 8 * np.sin(np.arange(days) * 2 * np.pi / 7) + rng.normal(0, 4, days))


class TestItRefusesWhatCannotBeForecast:
    def test_the_stock_level_series_that_fooled_it(self) -> None:
        """The actual failure this module exists for.

        Derived from a real sheet: the number sat unchanged for months, then jumped. The
        tool read that as demand and printed reorder points to the unit.
        """
        series = np.zeros(227)
        series[60] = 6000.0
        series[180] = 5200.0

        verdict = diagnose(series)

        assert verdict.shape is Shape.NOT_DEMAND
        assert not verdict.usable
        assert "stock level" in verdict.message
        assert verdict.flat_share > 0.95

    def test_a_completely_constant_series(self) -> None:
        verdict = diagnose(np.full(200, 37.0))
        assert verdict.shape is Shape.NOT_DEMAND
        assert not verdict.usable

    def test_too_little_history_says_how_much_is_needed(self) -> None:
        verdict = diagnose(smooth(days=20))
        assert verdict.shape is Shape.TOO_SHORT
        assert not verdict.usable
        assert str(MIN_OBSERVATIONS) in verdict.message

    @pytest.mark.parametrize("days", [0, 1, 5, MIN_OBSERVATIONS - 1])
    def test_short_series_never_raise(self, days: int) -> None:
        """Including empty. A crash here would be the same bug in a different coat."""
        assert not diagnose(np.zeros(days)).usable


class TestItNamesWhatItCanForecast:
    def test_steady_demand_passes_cleanly(self) -> None:
        verdict = diagnose(smooth())
        assert verdict.shape is Shape.SMOOTH
        assert verdict.usable
        assert not verdict.needs_a_different_model
        assert verdict.message == "Steady daily demand. Good to forecast."

    def test_the_projects_own_sample_data_passes(self) -> None:
        """If the generated demo data failed its own check, the check would be wrong."""
        frame = generate_sales(days=365, seed=11)
        for _, group in frame.groupby("sku"):
            verdict = diagnose(group["units"].to_numpy(dtype=float))
            assert verdict.usable, f"the demo data must pass: {verdict.message}"

    def test_intermittent_demand_is_usable_but_flagged(self) -> None:
        """Real demand, wrong model. Forecast it, but say so."""
        rng = np.random.default_rng(5)
        series = np.where(rng.random(200) < 0.25, rng.integers(1, 9, 200), 0).astype(float)

        verdict = diagnose(series)

        assert verdict.shape is Shape.INTERMITTENT
        assert verdict.usable, "lumpy demand is still demand"
        assert verdict.needs_a_different_model
        assert "intermittent" in verdict.message.lower()

    def test_a_bulk_movement_is_flagged_without_blocking(self) -> None:
        series = smooth()
        series[100] = series.mean() * 60

        verdict = diagnose(series)

        assert verdict.usable
        assert verdict.spike_days >= 1
        assert "correction" in verdict.message


class TestTheNumbersItReports:
    def test_zero_and_flat_shares_are_measured_not_guessed(self) -> None:
        series = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 7.0, 0.0, 0.0, 3.0, 3.0] * 6)
        verdict = diagnose(series)
        assert verdict.zero_share == pytest.approx(0.5)
        assert 0.0 < verdict.flat_share < 1.0

    def test_spikes_are_judged_against_active_days_not_zeros(self) -> None:
        """A mostly-idle product must not be called spiky just for being idle."""
        series = np.zeros(200)
        series[::4] = 10.0
        assert diagnose(series).spike_days == 0
