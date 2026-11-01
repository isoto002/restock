"""Model selection, and Croston's method underneath it.

The claim being tested is not "Croston works". It is that letting the models COMPETE picks
a better one than assuming Holt-Winters, specifically on the lumpy demand that assumption
was wrong for.
"""

from __future__ import annotations

import numpy as np
import pytest

from restock.forecast import CrostonFit, croston
from restock.metrics import accuracy
from restock.select import (
    INTERMITTENT,
    NAIVE,
    SEASONAL,
    select_model,
)


def seasonal_series(days: int = 200, seed: int = 3) -> np.ndarray:
    """Sells every day with a clear weekly shape. Holt-Winters territory."""
    rng = np.random.default_rng(seed)
    week = np.sin(np.arange(days) * 2 * np.pi / 7)
    return np.maximum(0, 60 + 25 * week + rng.normal(0, 3, days))


def lumpy_series(days: int = 200, seed: int = 9, rate: float = 0.12) -> np.ndarray:
    """Sells on roughly one day in eight, in batches. Croston territory."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(days) < rate, rng.integers(20, 60, days), 0).astype(float)


class TestCroston:
    def test_it_estimates_size_and_interval_separately(self) -> None:
        """Demand of 10 every 5th day: size near 10, interval near 5."""
        series = np.zeros(100)
        series[::5] = 10.0

        fit = croston(series, alpha=0.3)

        assert fit.size == pytest.approx(10.0, abs=0.5)
        assert fit.interval == pytest.approx(5.0, abs=0.5)

    def test_the_rate_is_demand_per_period_not_per_order(self) -> None:
        """10 units every 5 days is 2 a day, not 10."""
        series = np.zeros(100)
        series[::5] = 10.0
        assert croston(series, alpha=0.3, debias=False).rate == pytest.approx(2.0, abs=0.2)

    def test_the_debias_correction_lowers_the_rate(self) -> None:
        """Classic Croston is known to run high. The correction is on by default."""
        series = lumpy_series()
        assert croston(series, debias=True).rate < croston(series, debias=False).rate
        assert croston(series).debiased

    def test_a_series_with_no_demand_gives_zero_not_a_divide_error(self) -> None:
        fit = croston(np.zeros(50))
        assert isinstance(fit, CrostonFit)
        assert fit.rate == 0.0

    def test_every_period_gets_the_same_number(self) -> None:
        """Croston forecasts a rate, not a shape. Pretending otherwise would be a lie."""
        forecast = croston(lumpy_series()).predict(10)
        assert len(forecast) == 10
        assert len(set(forecast.tolist())) == 1

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
    def test_it_rejects_an_impossible_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            croston(lumpy_series(), alpha=alpha)


class TestSelectionPicksTheRightModel:
    def test_seasonal_demand_picks_the_seasonal_model(self) -> None:
        choice = select_model(seasonal_series())
        assert choice.best.name == SEASONAL
        assert choice.beat_the_baseline

    def test_lumpy_demand_does_not_pick_the_seasonal_model(self) -> None:
        """The whole reason this module exists.

        Holt-Winters on one-day-in-eight demand fits the gaps. It must not win here.
        """
        choice = select_model(lumpy_series())
        assert choice.best.name != SEASONAL

    def test_selection_beats_always_using_holt_winters_on_lumpy_demand(self) -> None:
        """Measured, not asserted: the chosen model scores better than the old default."""
        series = lumpy_series()
        choice = select_model(series)
        seasonal = next(c for c in choice.candidates if c.name == SEASONAL)

        assert choice.best.mae < seasonal.mae, (
            f"{choice.best.name} ({choice.best.mae:.2f}) should beat "
            f"{SEASONAL} ({seasonal.mae:.2f}) on intermittent demand"
        )


class TestTheComparisonIsHonest:
    def test_every_candidate_is_reported_best_first(self) -> None:
        choice = select_model(seasonal_series())
        names = [c.name for c in choice.candidates]
        assert set(names) == {NAIVE, SEASONAL, INTERMITTENT}
        assert [c.mae for c in choice.candidates] == sorted(c.mae for c in choice.candidates)

    def test_the_winner_is_refitted_on_the_whole_series(self) -> None:
        """The scored model never saw the recent weeks, which are the ones that matter."""
        choice = select_model(seasonal_series())
        assert choice.best.predict is not None
        assert len(choice.best.predict(14)) == 14

    def test_a_model_that_cannot_fit_is_recorded_not_raised(self) -> None:
        """One broken candidate must not take the whole selection down."""
        choice = select_model(np.zeros(100))
        assert choice.best.predict is not None

    def test_it_refuses_a_series_too_short_to_score(self) -> None:
        with pytest.raises(ValueError, match="hold"):
            select_model(seasonal_series(days=20))

    def test_flagging_when_nothing_beat_the_average(self) -> None:
        """Pure noise: no model should claim a win over the mean."""
        rng = np.random.default_rng(1)
        choice = select_model(rng.normal(50, 10, 200))
        if choice.best.name == NAIVE:
            assert not choice.beat_the_baseline


class TestScoring:
    def test_mae_counts_zero_periods_and_mape_does_not(self) -> None:
        """Why selection ranks on MAE. On lumpy demand MAPE sees almost nothing."""
        actual = np.array([0.0, 0.0, 0.0, 10.0])
        scored = accuracy(actual=actual, forecast=np.full(4, 2.5))

        assert scored.periods_skipped == 3
        assert scored.periods_scored == 1
        assert scored.mae == pytest.approx((2.5 + 2.5 + 2.5 + 7.5) / 4)
