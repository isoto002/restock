"""Synthetic sales history, so the repo demos with no real data.

Real sales data is somebody's property. Everything here is generated from a seed, which
means the examples in the README are reproducible and nobody has to hand over a file to
try the tool.

The generated series deliberately carries the three signals a retail forecaster has to
cope with: a slow upward trend, a weekly cycle (weekends sell differently from Tuesdays),
and a Q4 lift. `tests/test_simulate.py` asserts those signals are actually present — a
generator that quietly produced flat noise would let every forecast test pass while the
demo looked obviously wrong to anyone who knows the domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Multiplier applied to each weekday, Monday first. Weekends run hotter.
WEEKDAY_SHAPE: tuple[float, ...] = (0.85, 0.90, 0.95, 1.00, 1.25, 1.40, 1.15)

#: Months counted as peak season, and the multiplier applied across them.
PEAK_MONTHS: frozenset[int] = frozenset({11, 12})
PEAK_MULTIPLIER = 1.6


@dataclass(frozen=True)
class SkuProfile:
    """How one product behaves.

    Attributes:
        sku: The product identifier as it appears in the output.
        base_units: Average units per day before any seasonal shaping is applied.
        trend_per_day: Units added per day of elapsed time. Negative means decline.
        noise: Standard deviation of the daily noise, as a fraction of the shaped mean.
    """

    sku: str
    base_units: float
    trend_per_day: float = 0.0
    noise: float = 0.15


DEFAULT_PROFILES: tuple[SkuProfile, ...] = (
    SkuProfile("WIDGET-BLUE", base_units=40.0, trend_per_day=0.05, noise=0.18),
    SkuProfile("WIDGET-RED", base_units=22.0, trend_per_day=0.00, noise=0.25),
    SkuProfile("GADGET-PRO", base_units=12.0, trend_per_day=0.03, noise=0.30),
    SkuProfile("GADGET-MINI", base_units=65.0, trend_per_day=-0.04, noise=0.12),
)


def _shaped_mean(profile: SkuProfile, dates: pd.DatetimeIndex) -> np.ndarray:
    """Expected units per day, before noise, in units.

    Combines the SKU's baseline, its linear trend, the weekday shape and the peak-season
    lift. Kept separate from the noise so the tests can assert the shape directly.
    """
    elapsed = np.arange(len(dates), dtype=float)
    level = profile.base_units + profile.trend_per_day * elapsed

    weekday = np.array([WEEKDAY_SHAPE[d] for d in dates.day_of_week], dtype=float)
    peak = np.where(dates.month.isin(PEAK_MONTHS), PEAK_MULTIPLIER, 1.0)

    return np.asarray(level * weekday * peak, dtype=float)


def generate_sales(
    *,
    start: str = "2024-01-01",
    days: int = 730,
    profiles: tuple[SkuProfile, ...] = DEFAULT_PROFILES,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a daily sales history.

    Args:
        start: First date in the series, ISO format.
        days: How many consecutive days to generate. Two years by default, because
            Holt-Winters needs at least two full seasonal cycles to fit.
        profiles: The products to generate. Defaults to four with different shapes.
        seed: Seeds the noise, so the same arguments always give the same history.

    Returns:
        A frame of `date` (datetime64), `sku` (str) and `units` (int), sorted by SKU then
        date. Units are never negative: demand that would go below zero is clipped, which
        is also what happens in real life when you cannot sell a negative quantity.

    Raises:
        ValueError: If `days` is below 1 or `profiles` is empty.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    if not profiles:
        raise ValueError("at least one SkuProfile is required")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")

    frames = []
    for profile in profiles:
        mean = _shaped_mean(profile, dates)
        noise = rng.normal(loc=1.0, scale=profile.noise, size=days)
        units = np.clip(np.rint(mean * noise), 0, None).astype(int)
        frames.append(pd.DataFrame({"date": dates, "sku": profile.sku, "units": units}))

    return pd.concat(frames, ignore_index=True).sort_values(["sku", "date"]).reset_index(drop=True)
