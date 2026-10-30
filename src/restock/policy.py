"""When to reorder, and how much cover to hold.

Two numbers decide whether a warehouse stocks out:

* **Safety stock** — the buffer that absorbs the two things that actually go wrong.
  Demand varies, and so does the supplier. Holding buffer for demand alone is the
  classic mistake: a supplier whose lead time swings between one and three weeks will
  stock you out no matter how steady your demand is.

      safety stock = z * sqrt( lead_time * demand_sd^2  +  demand_mean^2 * lead_time_sd^2 )
                               \\_____ demand varies _____/   \\___ supplier varies ___/

* **Reorder point** — the stock level that should trigger a purchase order. It covers
  expected demand across the lead time, plus that buffer.

      reorder point = demand_mean * lead_time + safety stock

`z` comes from the service level: the share of replenishment cycles you are willing to
get through without stocking out. It is a business decision, not a statistical one, and
it is deliberately a required argument rather than a default — 95% is a choice somebody
should make on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

#: Common service levels and their z-scores, for reference. Any level in (0, 1) works;
#: these are the ones that appear in practice.
COMMON_SERVICE_LEVELS: dict[float, float] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.98: 2.0537,
    0.99: 2.3263,
}


@dataclass(frozen=True)
class ReorderPolicy:
    """What to do about one SKU.

    Attributes:
        safety_stock: Buffer held against variability, in units.
        reorder_point: Stock level that should trigger a purchase order, in units.
        demand_over_lead_time: Expected demand across one lead time, in units.
        service_level: The service level this policy was built for, 0 to 1.
    """

    safety_stock: float
    reorder_point: float
    demand_over_lead_time: float
    service_level: float


def service_level_z(service_level: float) -> float:
    """The z-score for a service level.

    Args:
        service_level: Share of cycles to get through without a stockout, strictly
            between 0 and 1.

    Raises:
        ValueError: If outside the open interval (0, 1). A service level of exactly 1
            means never stocking out under any circumstances, which needs infinite
            buffer — so it is refused rather than returning infinity.
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError(
            f"service_level must be between 0 and 1 exclusive, got {service_level}. "
            "A service level of exactly 1 would require infinite safety stock."
        )
    return float(NormalDist().inv_cdf(service_level))


def reorder_policy(
    *,
    demand_mean: float,
    demand_sd: float,
    lead_time_days: float,
    service_level: float,
    lead_time_sd: float = 0.0,
) -> ReorderPolicy:
    """Work out the safety stock and reorder point for one SKU.

    Args:
        demand_mean: Average demand per day, in units.
        demand_sd: Standard deviation of daily demand, in units.
        lead_time_days: Average days between placing an order and receiving it.
        service_level: Share of cycles to get through without a stockout, 0 to 1 exclusive.
        lead_time_sd: Standard deviation of the lead time, in days. Defaults to 0, meaning
            a perfectly reliable supplier — rare, and worth measuring before assuming.

    Returns:
        A `ReorderPolicy`.

    Raises:
        ValueError: If any figure is negative, or `service_level` is outside (0, 1).
    """
    _validate(demand_mean, demand_sd, lead_time_days, lead_time_sd)

    z = service_level_z(service_level)
    demand_variance = lead_time_days * demand_sd**2
    supply_variance = (demand_mean**2) * (lead_time_sd**2)
    safety_stock = z * math.sqrt(demand_variance + supply_variance)

    demand_over_lead_time = demand_mean * lead_time_days
    return ReorderPolicy(
        safety_stock=safety_stock,
        reorder_point=demand_over_lead_time + safety_stock,
        demand_over_lead_time=demand_over_lead_time,
        service_level=service_level,
    )


def days_of_cover(on_hand: float, demand_mean: float) -> float:
    """How many days the stock on hand will last at the current demand rate.

    Returns `inf` when demand is zero — stock that nothing is consuming lasts forever,
    which is the honest answer and reads better in a sorted at-risk table than a crash.

    Raises:
        ValueError: If either figure is negative.
    """
    if on_hand < 0:
        raise ValueError(f"on_hand cannot be negative, got {on_hand}")
    if demand_mean < 0:
        raise ValueError(f"demand_mean cannot be negative, got {demand_mean}")
    if demand_mean == 0:
        return float("inf")
    return on_hand / demand_mean


def _validate(
    demand_mean: float, demand_sd: float, lead_time_days: float, lead_time_sd: float
) -> None:
    """Reject negative inputs. See `reorder_policy`."""
    for name, value in (
        ("demand_mean", demand_mean),
        ("demand_sd", demand_sd),
        ("lead_time_days", lead_time_days),
        ("lead_time_sd", lead_time_sd),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")
