"""Forecasting, written out rather than imported.

`statsmodels` already has Holt-Winters and in production you should usually use it. It is
written out here on purpose: the point of this repo is to show the method, and an import
would hide exactly the part worth showing.

Two forecasters live here:

* `moving_average` — the baseline. Every forecast should be compared against something
  naive, because a model that cannot beat a seven-day average is not earning its keep.
* `holt_winters` — triple exponential smoothing: level, trend and a repeating seasonal
  cycle, each updated as new demand arrives.

The additive seasonal form is used, which suits demand that swings by roughly the same
NUMBER of units each cycle. Demand that swings by a PERCENTAGE wants the multiplicative
form instead; that is a deliberate omission, noted in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SEASONAL_CYCLES = 2


class InsufficientHistoryError(ValueError):
    """Raised when there is not enough history to fit the requested model.

    Holt-Winters needs at least two complete seasonal cycles: one to estimate the initial
    seasonal shape and a second to estimate the trend between them. With less it will
    still return numbers, and they will be meaningless — so it refuses instead.
    """


@dataclass(frozen=True)
class HoltWintersFit:
    """A fitted model and the pieces it was built from.

    Attributes:
        fitted: One-step-ahead predictions aligned with the input series, in units.
        level: Final level estimate, in units.
        trend: Final trend estimate, in units per period.
        season: Final seasonal offsets, one per period of the cycle, in units.
        season_length: Periods in one seasonal cycle.
        periods_fitted: How many observations the model was fitted on. Needed so the
            forecast picks up the seasonal cycle at the right phase — see `predict`.
    """

    fitted: np.ndarray
    level: float
    trend: float
    season: np.ndarray
    season_length: int
    periods_fitted: int

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next `horizon` periods, in units.

        The seasonal offset has to continue the cycle from where the fit ended, not
        restart it. If a model fitted on 730 days of weekly data forecast day 731 using
        Monday's offset, every prediction would carry the wrong day's shape — and with a
        real weekday pattern that is enough to make the model lose to a flat average.
        The phase of the next period is `periods_fitted % season_length`, so step `h`
        lands on `(periods_fitted + h - 1) % season_length`.

        Raises:
            ValueError: If `horizon` is below 1.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        steps = np.arange(1, horizon + 1)
        phase = (self.periods_fitted + steps - 1) % self.season_length
        return np.asarray(self.level + steps * self.trend + self.season[phase], dtype=float)


def moving_average(series: np.ndarray, window: int = 7) -> float:
    """Mean of the last `window` observations, in units — the baseline to beat.

    Raises:
        ValueError: If `window` is below 1.
        InsufficientHistoryError: If the series is shorter than `window`.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if len(series) < window:
        raise InsufficientHistoryError(
            f"moving_average needs at least {window} observations, got {len(series)}"
        )
    return float(np.mean(series[-window:]))


def _initial_state(series: np.ndarray, season_length: int) -> tuple[float, float, np.ndarray]:
    """Seed level, trend and seasonal offsets from the first two cycles.

    The level starts at the first cycle's mean and the trend at the per-period gap between
    the first two cycle means.

    The seasonal offsets are measured against a DE-TRENDED first cycle, not against the
    flat mean. That detail matters more than it looks: a series that rises steadily within
    each cycle would otherwise have that rise baked into the seasonal offsets, so the model
    double-counts it as "Mondays are low, Thursdays are high" and the forecast drifts low
    forever. Measuring each point against where the trend line says it should be leaves a
    pure trend with seasonal offsets of exactly zero, which is the correct answer.

    The first cycle's mean describes its CENTRE, at position (season_length - 1) / 2, so
    that is the point the trend line is anchored at.
    """
    first = series[:season_length]
    second = series[season_length : 2 * season_length]

    level = float(np.mean(first))
    trend = float((np.mean(second) - level) / season_length)

    centre = (season_length - 1) / 2.0
    trend_line = level + (np.arange(season_length, dtype=float) - centre) * trend
    season = np.asarray(first - trend_line, dtype=float)
    return level, trend, season


def holt_winters(
    series: np.ndarray,
    season_length: int,
    *,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.2,
) -> HoltWintersFit:
    """Fit additive Holt-Winters to `series`.

    Args:
        series: Observed demand per period, in units, oldest first.
        season_length: Periods in one seasonal cycle — 7 for daily data with a weekly cycle.
        alpha: Level smoothing, 0 to 1. Higher reacts faster and is noisier.
        beta: Trend smoothing, 0 to 1.
        gamma: Seasonal smoothing, 0 to 1.

    Returns:
        A `HoltWintersFit` carrying the fitted values and the final state.

    Raises:
        ValueError: If `season_length` is below 2 or a smoothing factor is outside 0 to 1.
        InsufficientHistoryError: If the series is shorter than two full seasonal cycles.
    """
    _validate_holt_winters(series, season_length, alpha, beta, gamma)

    level, trend, season = _initial_state(series, season_length)
    fitted = np.empty(len(series), dtype=float)

    for t, observed in enumerate(series):
        seasonal = season[t % season_length]
        fitted[t] = level + trend + seasonal

        previous_level = level
        level = alpha * (observed - seasonal) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
        season[t % season_length] = gamma * (observed - level) + (1 - gamma) * seasonal

    return HoltWintersFit(fitted, level, trend, season, season_length, len(series))


def _validate_holt_winters(
    series: np.ndarray, season_length: int, alpha: float, beta: float, gamma: float
) -> None:
    """Reject arguments that would produce meaningless output. See `holt_winters`."""
    if season_length < 2:
        raise ValueError(f"season_length must be at least 2, got {season_length}")

    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")

    required = MIN_SEASONAL_CYCLES * season_length
    if len(series) < required:
        raise InsufficientHistoryError(
            f"holt_winters needs {MIN_SEASONAL_CYCLES} full cycles "
            f"({required} observations at season_length={season_length}), got {len(series)}"
        )
