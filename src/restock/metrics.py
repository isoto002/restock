"""Forecast accuracy.

Two numbers, because they answer different questions:

* **MAPE** — how far off are we, on average, as a percentage? Good for comparing SKUs
  that sell at wildly different volumes.
* **Bias** — are we consistently over or under? A forecast can have a respectable MAPE
  and still be biased low every single week, which quietly starves the warehouse. MAPE
  alone will never tell you that, which is why both are here.

MAPE divides by actual demand, so a zero-demand period makes it infinite. That is a real
property of the metric, not a bug to paper over, so the behaviour is explicit: zero-demand
periods are excluded from the calculation and counted, and the count is returned alongside
the number so a caller can see how much was skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Accuracy:
    """Forecast accuracy over a set of periods.

    Attributes:
        mape: Mean absolute percentage error, as a percentage. `nan` when every period
            had zero demand and nothing could be measured.
        bias: Mean signed error in units. Positive means the forecast ran high.
        periods_scored: Periods that contributed to `mape`.
        periods_skipped: Periods excluded from `mape` because actual demand was zero.
    """

    mape: float
    bias: float
    periods_scored: int
    periods_skipped: int


def accuracy(actual: np.ndarray, forecast: np.ndarray) -> Accuracy:
    """Score `forecast` against `actual`.

    Args:
        actual: Observed demand per period, in units.
        forecast: Predicted demand for the same periods, in units.

    Returns:
        An `Accuracy`. See the module docstring for the zero-demand policy.

    Raises:
        ValueError: If the two arrays differ in length, or are empty.
    """
    if len(actual) != len(forecast):
        raise ValueError(
            f"actual and forecast must be the same length, got {len(actual)} and {len(forecast)}"
        )
    if len(actual) == 0:
        raise ValueError("cannot score an empty series")

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    bias = float(np.mean(forecast - actual))

    scorable = actual != 0
    skipped = int(np.sum(~scorable))
    if not scorable.any():
        return Accuracy(mape=float("nan"), bias=bias, periods_scored=0, periods_skipped=skipped)

    errors = np.abs((actual[scorable] - forecast[scorable]) / actual[scorable])
    return Accuracy(
        mape=float(np.mean(errors) * 100.0),
        bias=bias,
        periods_scored=int(np.sum(scorable)),
        periods_skipped=skipped,
    )
