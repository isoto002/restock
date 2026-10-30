"""The generator has to produce data that actually looks like retail demand.

This is the test file that is easy to skip and expensive to skip. If the generator
quietly produced flat noise, every forecasting test in this repo would still pass — they
run on their own constructed series — while the demo on the front page looked obviously
wrong to anyone who has worked a warehouse. So the properties the generator CLAIMS are
asserted here directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from restock.simulate import DEFAULT_PROFILES, PEAK_MONTHS, SkuProfile, generate_sales


class TestShape:
    def test_returns_one_row_per_sku_per_day(self) -> None:
        frame = generate_sales(days=30, profiles=DEFAULT_PROFILES)
        assert len(frame) == 30 * len(DEFAULT_PROFILES)

    def test_has_the_documented_columns_and_types(self) -> None:
        frame = generate_sales(days=10)
        assert list(frame.columns) == ["date", "sku", "units"]
        assert pd.api.types.is_datetime64_any_dtype(frame["date"])
        assert pd.api.types.is_integer_dtype(frame["units"])

    def test_demand_is_never_negative(self) -> None:
        """You cannot sell a negative quantity, so noise is clipped at zero."""
        noisy = (SkuProfile("JITTERY", base_units=2.0, noise=3.0),)
        frame = generate_sales(days=400, profiles=noisy, seed=99)
        assert frame["units"].min() >= 0

    def test_rejects_a_zero_day_range(self) -> None:
        with pytest.raises(ValueError, match="days must be at least 1"):
            generate_sales(days=0)

    def test_rejects_an_empty_profile_list(self) -> None:
        with pytest.raises(ValueError, match="at least one SkuProfile"):
            generate_sales(profiles=())


class TestReproducibility:
    def test_the_same_seed_gives_the_same_history(self) -> None:
        """The README's numbers have to be reproducible by whoever reads it."""
        first = generate_sales(days=60, seed=42)
        second = generate_sales(days=60, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_a_different_seed_gives_different_noise(self) -> None:
        first = generate_sales(days=60, seed=1)
        second = generate_sales(days=60, seed=2)
        assert not first["units"].equals(second["units"])


class TestSignalsArePresent:
    """The three properties the module docstring promises."""

    def test_weekly_seasonality_is_real(self) -> None:
        """Saturday carries the highest weekday multiplier, Monday the lowest.

        Averaged over two years the noise cancels, so the ordering must survive.
        """
        frame = generate_sales(days=730, profiles=(DEFAULT_PROFILES[0],), seed=3)
        by_weekday = frame.groupby(frame["date"].dt.day_of_week)["units"].mean()

        assert by_weekday.idxmax() == 5, "Saturday should be the busiest day"
        assert by_weekday.idxmin() == 0, "Monday should be the quietest day"

    def test_peak_season_lifts_demand(self) -> None:
        """November and December carry a 1.6x multiplier, so they must sell more."""
        frame = generate_sales(days=730, profiles=(DEFAULT_PROFILES[0],), seed=3)
        is_peak = frame["date"].dt.month.isin(PEAK_MONTHS)

        peak_mean = frame.loc[is_peak, "units"].mean()
        rest_mean = frame.loc[~is_peak, "units"].mean()
        assert peak_mean > rest_mean * 1.3

    def test_a_rising_trend_actually_rises(self) -> None:
        """Compare the first and last 90 days, well clear of the Q4 lift either side."""
        rising = (SkuProfile("CLIMBER", base_units=20.0, trend_per_day=0.1, noise=0.05),)
        frame = generate_sales(start="2024-01-01", days=365, profiles=rising, seed=5)

        first_quarter = frame.head(90)["units"].mean()
        last_quarter = frame.tail(90)["units"].mean()
        assert last_quarter > first_quarter

    def test_a_falling_trend_actually_falls(self) -> None:
        falling = (SkuProfile("FADER", base_units=60.0, trend_per_day=-0.1, noise=0.05),)
        frame = generate_sales(start="2024-01-01", days=365, profiles=falling, seed=5)

        assert frame.tail(90)["units"].mean() < frame.head(90)["units"].mean()
