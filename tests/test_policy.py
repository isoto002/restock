"""Reorder maths, checked against values worked out by hand.

Every expected number in this file was calculated on paper first. That is deliberate: a
test that asserts whatever the code happened to print proves only that the code is
consistent with itself.
"""

from __future__ import annotations

import math

import pytest

from restock.policy import COMMON_SERVICE_LEVELS, days_of_cover, reorder_policy, service_level_z


class TestServiceLevelZ:
    @pytest.mark.parametrize(("level", "expected"), COMMON_SERVICE_LEVELS.items())
    def test_matches_published_z_scores(self, level: float, expected: float) -> None:
        """The z-scores in any inventory textbook, to four decimal places."""
        assert service_level_z(level) == pytest.approx(expected, abs=1e-4)

    def test_fifty_percent_is_zero(self) -> None:
        """Half the cycles needs no buffer at all — the distribution's midpoint."""
        assert service_level_z(0.50) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_levels_outside_the_open_interval(self, level: float) -> None:
        """A service level of exactly 1 would need infinite buffer, so it is refused."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            service_level_z(level)


class TestReorderPolicy:
    def test_demand_variability_only(self) -> None:
        """By hand: z=1.6449, sqrt(9 * 20^2) = 60, so SS = 1.6449 * 60 = 98.69.

        Demand over the lead time is 100 * 9 = 900, so the reorder point is 998.69.
        """
        policy = reorder_policy(
            demand_mean=100.0, demand_sd=20.0, lead_time_days=9.0, service_level=0.95
        )
        assert policy.safety_stock == pytest.approx(98.69, abs=0.01)
        assert policy.demand_over_lead_time == pytest.approx(900.0)
        assert policy.reorder_point == pytest.approx(998.69, abs=0.01)

    def test_unreliable_supplier_dominates_the_buffer(self) -> None:
        """By hand: 9*400 = 3600 from demand, 100^2 * 2^2 = 40000 from the supplier.

        sqrt(43600) = 208.8061, so SS = 1.6448536 * 208.8061 = 343.456. The supplier term
        is more than ten times the demand term, which is the whole reason it is in the
        formula — buffering demand alone would under-stock by roughly 245 units.

        Use the full z-score here, not the four-decimal 1.6449 from the table: rounding it
        shifts the answer by about 0.01 and the tolerance below is tighter than that.
        """
        policy = reorder_policy(
            demand_mean=100.0,
            demand_sd=20.0,
            lead_time_days=9.0,
            service_level=0.95,
            lead_time_sd=2.0,
        )
        assert policy.safety_stock == pytest.approx(343.456, abs=0.01)

    def test_no_variability_needs_no_buffer(self) -> None:
        """Perfectly steady demand and a perfectly reliable supplier: SS is zero."""
        policy = reorder_policy(
            demand_mean=50.0, demand_sd=0.0, lead_time_days=5.0, service_level=0.99
        )
        assert policy.safety_stock == pytest.approx(0.0)
        assert policy.reorder_point == pytest.approx(250.0)

    def test_zero_lead_time_leaves_only_the_supplier_term(self) -> None:
        """Instant replenishment: nothing accrues over the lead time, so ROP is just SS."""
        policy = reorder_policy(
            demand_mean=30.0, demand_sd=10.0, lead_time_days=0.0, service_level=0.95
        )
        assert policy.demand_over_lead_time == pytest.approx(0.0)
        assert policy.safety_stock == pytest.approx(0.0)
        assert policy.reorder_point == pytest.approx(0.0)

    def test_higher_service_level_raises_the_buffer(self) -> None:
        """A monotonic relationship the maths must preserve."""
        lower = reorder_policy(
            demand_mean=100.0, demand_sd=20.0, lead_time_days=9.0, service_level=0.90
        )
        higher = reorder_policy(
            demand_mean=100.0, demand_sd=20.0, lead_time_days=9.0, service_level=0.99
        )
        assert higher.safety_stock > lower.safety_stock

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"demand_mean": -1.0, "demand_sd": 1.0, "lead_time_days": 1.0},
            {"demand_mean": 1.0, "demand_sd": -1.0, "lead_time_days": 1.0},
            {"demand_mean": 1.0, "demand_sd": 1.0, "lead_time_days": -1.0},
        ],
    )
    def test_rejects_negative_inputs(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            reorder_policy(service_level=0.95, **kwargs)


class TestDaysOfCover:
    def test_divides_stock_by_daily_demand(self) -> None:
        assert days_of_cover(on_hand=250.0, demand_mean=25.0) == pytest.approx(10.0)

    def test_zero_demand_lasts_forever(self) -> None:
        """Stock nothing is consuming never runs out — infinity sorts last, which is right."""
        assert math.isinf(days_of_cover(on_hand=100.0, demand_mean=0.0))

    def test_no_stock_is_zero_days(self) -> None:
        assert days_of_cover(on_hand=0.0, demand_mean=10.0) == pytest.approx(0.0)

    def test_rejects_negative_stock(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            days_of_cover(on_hand=-1.0, demand_mean=10.0)
