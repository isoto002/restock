"""Accuracy metrics, checked against arithmetic done by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from restock.metrics import accuracy


class TestMape:
    def test_matches_a_hand_calculation(self) -> None:
        """Errors of 10/100, 10/200 and 0/300 average to 0.05, so MAPE is 5%."""
        result = accuracy(
            actual=np.array([100.0, 200.0, 300.0]),
            forecast=np.array([110.0, 190.0, 300.0]),
        )
        assert result.mape == pytest.approx(5.0)

    def test_a_perfect_forecast_scores_zero(self) -> None:
        series = np.array([5.0, 10.0, 15.0])
        assert accuracy(actual=series, forecast=series).mape == pytest.approx(0.0)


class TestBias:
    def test_is_zero_when_errors_cancel(self) -> None:
        """Over by 10 then under by 10: accurate on average, which is what bias measures."""
        result = accuracy(
            actual=np.array([100.0, 100.0]),
            forecast=np.array([110.0, 90.0]),
        )
        assert result.bias == pytest.approx(0.0)

    def test_catches_a_forecast_that_is_always_low(self) -> None:
        """The failure MAPE hides: a steady 10% under-forecast reads as a tidy 10% MAPE.

        Bias is what tells you the warehouse is being starved every single period.
        """
        result = accuracy(
            actual=np.array([100.0, 100.0, 100.0]),
            forecast=np.array([90.0, 90.0, 90.0]),
        )
        assert result.mape == pytest.approx(10.0)
        assert result.bias == pytest.approx(-10.0)

    def test_positive_bias_means_the_forecast_ran_high(self) -> None:
        result = accuracy(actual=np.array([50.0]), forecast=np.array([60.0]))
        assert result.bias == pytest.approx(10.0)


class TestZeroDemandPolicy:
    def test_zero_demand_periods_are_skipped_and_counted(self) -> None:
        """MAPE divides by actual, so a zero-demand day would make it infinite.

        The day is excluded and reported rather than silently poisoning the number. Here
        only the second day is scorable: |100 - 110| / 100 = 10%.
        """
        result = accuracy(
            actual=np.array([0.0, 100.0]),
            forecast=np.array([10.0, 110.0]),
        )
        assert result.mape == pytest.approx(10.0)
        assert result.periods_scored == 1
        assert result.periods_skipped == 1

    def test_bias_still_counts_the_zero_days(self) -> None:
        """Bias does not divide by anything, so a zero-demand day is real information."""
        result = accuracy(
            actual=np.array([0.0, 100.0]),
            forecast=np.array([10.0, 110.0]),
        )
        assert result.bias == pytest.approx(10.0)

    def test_an_all_zero_series_gives_nan_rather_than_a_crash(self) -> None:
        """Nothing is measurable, and saying so beats inventing a number."""
        result = accuracy(actual=np.zeros(3), forecast=np.array([1.0, 2.0, 3.0]))
        assert math.isnan(result.mape)
        assert result.periods_scored == 0
        assert result.periods_skipped == 3


class TestValidation:
    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            accuracy(actual=np.array([1.0, 2.0]), forecast=np.array([1.0]))

    def test_rejects_an_empty_series(self) -> None:
        with pytest.raises(ValueError, match="empty series"):
            accuracy(actual=np.array([]), forecast=np.array([]))
