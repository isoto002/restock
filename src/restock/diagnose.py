"""Decide whether a series is demand at all, before anyone forecasts it.

This module exists because of a specific mistake. A real inventory sheet was loaded as
demand history and planned against, and it produced a confident reorder table to the unit.
The input was a record of STOCK LEVELS, not consumption: the median product's number moved
on 1 day out of 227. There was nothing to forecast, and nothing said so.

A forecaster that fits whatever it is handed is worse than no forecaster, because the output
looks identical either way. So every series is characterised first, and a series that cannot
support a forecast is refused by name rather than quietly fitted.

The verdict is also what model selection reads: smooth demand and intermittent demand want
different models, and that choice should follow from a measurement rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

#: Below this many observations there is not enough to fit and still hold a test window back.
MIN_OBSERVATIONS = 42

#: A series this constant is not demand. It is a stock level, a stale export, or a column of
#: totals — something whose value only changes when a person edits it.
MAX_FLAT_SHARE = 0.95

#: Above this share of zero days, demand is real but arrives in lumps. Holt-Winters assumes a
#: smooth seasonal shape and fits noise here; Croston's method is the usual answer.
INTERMITTENT_ZERO_SHARE = 0.50

#: A single day this many times the median is a correction or a bulk movement, not demand.
SPIKE_MULTIPLE = 20.0


class Shape(Enum):
    """What a series turned out to be."""

    SMOOTH = "smooth"
    INTERMITTENT = "intermittent"
    NOT_DEMAND = "not demand"
    TOO_SHORT = "too short"


@dataclass(frozen=True)
class Verdict:
    """What the series is, whether it can be forecast, and what to tell the person.

    Attributes:
        shape: Which bucket the series fell into.
        usable: False when forecasting it would produce a confident wrong answer.
        message: One sentence written for somebody who did not write this code.
        zero_share: Share of observations that are exactly zero.
        flat_share: Share of observations equal to the one before them.
        spike_days: Count of days far above the typical non-zero day.
    """

    shape: Shape
    usable: bool
    message: str
    zero_share: float
    flat_share: float
    spike_days: int

    @property
    def needs_a_different_model(self) -> bool:
        """True when the series is real demand but not the smooth kind Holt-Winters wants."""
        return self.shape is Shape.INTERMITTENT


def _shares(series: np.ndarray) -> tuple[float, float]:
    """Share of values that are zero, and share equal to their predecessor.

    An empty series reports zeros rather than NaN. This module's whole job is refusing to
    hand back numbers that mean nothing, so it should not produce one itself.
    """
    if len(series) == 0:
        return 0.0, 0.0
    zero = float(np.mean(series == 0))
    flat = float(np.mean(np.diff(series) == 0)) if len(series) > 1 else 1.0
    return zero, flat


def _spike_days(series: np.ndarray) -> int:
    """Days whose value dwarfs a typical active day.

    Measured against the median of the NON-ZERO values, so a mostly-idle product is not
    judged against its own zeros.
    """
    active = series[series > 0]
    if len(active) < 2:
        return 0
    typical = float(np.median(active))
    return 0 if typical <= 0 else int(np.sum(series > typical * SPIKE_MULTIPLE))


def diagnose(units: np.ndarray) -> Verdict:
    """Characterise one product's history before anything tries to forecast it.

    Args:
        units: Observations in order, one per period, zeros included. Zeros matter: a day
            with no sales is information, and dropping them is how intermittent demand gets
            mistaken for smooth demand.

    Returns:
        A `Verdict`. Check `usable` before forecasting, and `needs_a_different_model`
        before assuming Holt-Winters is the right fit.
    """
    series = np.asarray(units, dtype=float)
    zero_share, flat_share = _shares(series)
    spikes = _spike_days(series)

    def verdict(shape: Shape, usable: bool, message: str) -> Verdict:
        return Verdict(shape, usable, message, zero_share, flat_share, spikes)

    if len(series) < MIN_OBSERVATIONS:
        return verdict(
            Shape.TOO_SHORT,
            False,
            f"Only {len(series)} days of history. Forecasting seasonal demand needs at "
            f"least {MIN_OBSERVATIONS}: two full weekly cycles to learn from, plus four "
            "weeks held back to measure the error on.",
        )

    if flat_share >= MAX_FLAT_SHARE:
        return verdict(
            Shape.NOT_DEMAND,
            False,
            f"This value is unchanged on {flat_share:.0%} of days, so there is no demand "
            "signal in it. It looks like a stock level or an export that only updates when "
            "somebody edits it, rather than units sold per day.",
        )

    if zero_share >= INTERMITTENT_ZERO_SHARE:
        return verdict(
            Shape.INTERMITTENT,
            True,
            f"Demand is intermittent: nothing moves on {zero_share:.0%} of days. The "
            "forecast still runs, but a seasonal model is a poor fit for lumpy demand, so "
            "treat the error figure as the honest one and lean on safety stock.",
        )

    if spikes:
        return verdict(
            Shape.SMOOTH,
            True,
            f"Usable, with {spikes} day(s) far above the rest. A single enormous day is "
            "usually a correction or a bulk transfer rather than demand, and it will inflate "
            "both the forecast and the buffer.",
        )

    return verdict(Shape.SMOOTH, True, "Steady daily demand. Good to forecast.")
