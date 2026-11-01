"""Choose a forecasting model per product by testing them, not by assuming.

The first version of this project fitted Holt-Winters to everything. That is a seasonal
model, and it is a good one on a product that sells most days with a weekly rhythm. On a
product that sells on one day in twenty it fits the gaps between orders rather than the
orders, and it loses to a flat average.

Rather than pick by rule, every candidate is fitted on the earlier part of the history and
scored on a window it never saw. The one with the lowest error wins. A naive flat average is
always in the running, and it wins more often than people expect — which is the point of
keeping it there. A model that cannot beat the average has not earned the complexity.

Scored on MAE, not MAPE. MAPE is undefined when actual demand is zero, and on intermittent
demand most periods are zero; ranking models by it would judge them on a handful of days.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from restock.forecast import croston, holt_winters
from restock.metrics import accuracy

#: Periods held back to score on. Four weeks: long enough to cover the seasonal cycle a few
#: times, short enough to leave a usable amount to fit on.
DEFAULT_HOLDOUT = 28

#: A fitted model: anything that can forecast a horizon.
Forecaster = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class Candidate:
    """One model's result on the held-out window.

    Attributes:
        name: What to call it in front of a person.
        mae: Mean absolute error on the holdout, in units. Lower is better.
        mape: Percentage error on the holdout; `nan` when every held-out period was zero.
        predict: The model refitted on the WHOLE series, ready to forecast forward.
        failed: Why this model could not be fitted, or None if it was.
    """

    name: str
    mae: float
    mape: float
    predict: Forecaster | None
    failed: str | None = None


@dataclass(frozen=True)
class Selection:
    """The chosen model and everything it beat.

    Attributes:
        best: The winning candidate.
        candidates: Every candidate tried, best first. Kept so the page can show the
            comparison — a forecast you cannot interrogate is a forecast you cannot trust.
    """

    best: Candidate
    candidates: list[Candidate]

    @property
    def beat_the_baseline(self) -> bool:
        """True when the winner is not the flat average.

        Worth surfacing. If the average wins, the fancy model added nothing, and saying so
        is more useful than quietly shipping the fancy one anyway.
        """
        return self.best.name != NAIVE


NAIVE = "Flat average"
SEASONAL = "Holt-Winters (seasonal)"
INTERMITTENT = "Croston (intermittent)"


def _naive(train: np.ndarray) -> Forecaster:
    """The baseline every other model has to beat: tomorrow looks like the average."""
    mean = float(np.mean(train)) if len(train) else 0.0
    return lambda horizon: np.full(horizon, mean, dtype=float)


def _seasonal(train: np.ndarray, season_length: int) -> Forecaster:
    return holt_winters(train, season_length=season_length).predict


def _intermittent(train: np.ndarray) -> Forecaster:
    return croston(train).predict


def _score(
    name: str, build: Callable[[np.ndarray], Forecaster], series: np.ndarray, holdout: int
) -> Candidate:
    """Fit on the early part, score on the part held back, then refit on everything.

    Refitting on the full series matters: the scored model deliberately never saw the most
    recent weeks, and those are the ones that matter most for what to order tomorrow.
    """
    train, test = series[:-holdout], series[-holdout:]
    try:
        scored = accuracy(actual=test, forecast=build(train)(holdout))
        return Candidate(name, scored.mae, scored.mape, build(series))
    except Exception as exc:
        # Deliberately broad. A candidate that cannot be fitted on this particular series
        # is a RESULT — it scores infinitely badly and loses — not a reason to take the
        # whole selection down and leave the product with no forecast at all.
        return Candidate(name, float("inf"), float("nan"), None, failed=str(exc))


def select_model(
    series: np.ndarray, season_length: int = 7, holdout: int = DEFAULT_HOLDOUT
) -> Selection:
    """Try every model on one product's history and return the one that scored best.

    Args:
        series: Demand per period, zeros included.
        season_length: Periods in one seasonal cycle, for the seasonal candidate.
        holdout: Periods held back to score on.

    Returns:
        A `Selection`. `best.predict` is refitted on the whole series and ready to use.

    Raises:
        ValueError: If the series is too short to both fit and score on.
    """
    series = np.asarray(series, dtype=float)
    if len(series) <= holdout:
        raise ValueError(
            f"need more than {holdout} observations to hold {holdout} back for scoring, "
            f"got {len(series)}"
        )

    candidates = [
        _score(NAIVE, _naive, series, holdout),
        _score(SEASONAL, lambda s: _seasonal(s, season_length), series, holdout),
        _score(INTERMITTENT, _intermittent, series, holdout),
    ]
    ranked = sorted(candidates, key=lambda c: c.mae)
    return Selection(best=ranked[0], candidates=ranked)
