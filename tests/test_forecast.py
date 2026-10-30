"""Forecasting, verified against series whose correct answer is derivable by hand.

There is no published worked example bundled here. Instead the models are checked against
constructed series where the right answer follows from the arithmetic — a clean repeating
cycle must be reproduced, a pure linear trend must be extrapolated, and the initialisation
step is checked term by term. That is a stronger check than matching one textbook table,
because it pins behaviour across the whole method rather than at a single point.
"""

from __future__ import annotations

import numpy as np
import pytest

from restock.forecast import (
    InsufficientHistoryError,
    _initial_state,
    holt_winters,
    moving_average,
)


class TestMovingAverage:
    def test_averages_the_window(self) -> None:
        """Last three of 1..10 are 8, 9, 10 — mean 9."""
        series = np.arange(1, 11, dtype=float)
        assert moving_average(series, window=3) == pytest.approx(9.0)

    def test_ignores_history_outside_the_window(self) -> None:
        """A huge early spike must not move a short window."""
        series = np.array([1000.0, 1.0, 1.0, 1.0], dtype=float)
        assert moving_average(series, window=3) == pytest.approx(1.0)

    def test_refuses_a_series_shorter_than_the_window(self) -> None:
        with pytest.raises(InsufficientHistoryError, match="at least 5"):
            moving_average(np.array([1.0, 2.0]), window=5)

    def test_rejects_a_zero_window(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            moving_average(np.array([1.0]), window=0)


class TestInitialState:
    def test_seeds_level_trend_and_season_by_hand(self) -> None:
        """Two identical cycles of [10, 20, 30, 40].

        Level is the first cycle's mean, 25. Both cycles average 25, so the trend is
        (25 - 25) / 4 = 0. Each seasonal offset is that period's distance from 25, giving
        [-15, -5, 5, 15].
        """
        series = np.array([10, 20, 30, 40, 10, 20, 30, 40], dtype=float)
        level, trend, season = _initial_state(series, season_length=4)

        assert level == pytest.approx(25.0)
        assert trend == pytest.approx(0.0)
        assert season == pytest.approx(np.array([-15.0, -5.0, 5.0, 15.0]))

    def test_detects_a_rising_trend_between_cycles(self) -> None:
        """Second cycle averages 8 more than the first, over 4 periods: trend = 2."""
        series = np.array([10, 20, 30, 40, 18, 28, 38, 48], dtype=float)
        _, trend, _ = _initial_state(series, season_length=4)
        assert trend == pytest.approx(2.0)

    def test_a_pure_trend_has_no_seasonality(self) -> None:
        """0, 5, 10, 15, ... rises by 5 every period with no repeating shape at all.

        The seasonal offsets must come out as exactly zero. Measuring them against the
        flat cycle mean instead of the trend line gives [-7.5, -2.5, 2.5, 7.5] — the trend
        counted twice — and the forecast then drifts low forever.
        """
        series = np.arange(0, 40, 5, dtype=float)
        _, trend, season = _initial_state(series, season_length=4)

        assert trend == pytest.approx(5.0)
        assert season == pytest.approx(np.zeros(4), abs=1e-9)


class TestHoltWinters:
    def test_reproduces_a_clean_repeating_cycle(self) -> None:
        """A noiseless cycle repeated eight times must be forecast back almost exactly.

        This is the property that matters: if the model cannot recover a pattern that is
        literally repeated in front of it, nothing else it says is trustworthy.
        """
        cycle = [10.0, 20.0, 30.0, 40.0]
        series = np.array(cycle * 8, dtype=float)

        fit = holt_winters(series, season_length=4)
        predicted = fit.predict(4)

        assert predicted == pytest.approx(np.array(cycle), abs=1.0)

    def test_extrapolates_a_pure_linear_trend(self) -> None:
        """Flat seasonality, demand rising by 5 a day: the next day should be ~5 higher."""
        series = np.arange(0, 200, 5, dtype=float)

        fit = holt_winters(series, season_length=4, alpha=0.5, beta=0.5, gamma=0.1)
        predicted = fit.predict(1)

        assert predicted[0] == pytest.approx(series[-1] + 5.0, abs=2.0)

    def test_tracks_a_flat_series(self) -> None:
        """Constant demand forecasts as constant demand."""
        series = np.full(40, 100.0)
        fit = holt_winters(series, season_length=7)
        assert fit.predict(7) == pytest.approx(np.full(7, 100.0), abs=0.5)

    def test_refuses_fewer_than_two_seasonal_cycles(self) -> None:
        """The failure this guard exists for: it would return numbers, and they would lie."""
        series = np.arange(13, dtype=float)
        with pytest.raises(InsufficientHistoryError, match="2 full cycles"):
            holt_winters(series, season_length=7)

    def test_accepts_exactly_two_cycles(self) -> None:
        """The boundary is inclusive — two full cycles is enough to fit."""
        series = np.arange(14, dtype=float)
        fit = holt_winters(series, season_length=7)
        assert len(fit.fitted) == 14

    @pytest.mark.parametrize(("name", "value"), [("alpha", 1.5), ("beta", -0.1), ("gamma", 2.0)])
    def test_rejects_smoothing_factors_outside_zero_to_one(self, name: str, value: float) -> None:
        series = np.full(40, 10.0)
        with pytest.raises(ValueError, match=f"{name} must be between 0 and 1"):
            holt_winters(series, season_length=7, **{name: value})

    def test_rejects_a_season_length_below_two(self) -> None:
        with pytest.raises(ValueError, match="season_length must be at least 2"):
            holt_winters(np.full(40, 10.0), season_length=1)

    def test_predict_rejects_a_zero_horizon(self) -> None:
        fit = holt_winters(np.full(40, 10.0), season_length=7)
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            fit.predict(0)

    def test_predict_repeats_the_seasonal_cycle_beyond_one_season(self) -> None:
        """Asking for two cycles ahead must wrap the seasonal offsets, not run off the end."""
        fit = holt_winters(np.array([10.0, 20.0, 30.0, 40.0] * 8), season_length=4)
        predicted = fit.predict(8)
        assert predicted[:4] == pytest.approx(predicted[4:], abs=2.0)
