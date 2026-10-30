# Why Holt-Winters is written out instead of imported

`statsmodels` has `ExponentialSmoothing`. It is better tested than this, it fits the
smoothing parameters for you, and in a production system that is what you should reach
for. So the obvious question about this repo is why the method is written out by hand.

Three reasons, in order of honesty.

## 1. The method is the thing being shown

This repo exists to show that I can implement a forecasting algorithm, not that I can
call one. `from statsmodels.tsa.holtwinters import ExponentialSmoothing` is one line and
it hides the entire point.

## 2. Writing it out is what exposed the bug

The seasonal-phase bug described in the README is a good example of what you do not learn
from an import. The forecast was restarting the seasonal cycle instead of continuing it,
and finding that meant understanding what the seasonal array actually holds and which
index the next period maps to.

A library would have been correct from the start, and I would have learned nothing. The
version that was wrong, and then measurably right, taught more than the version that was
never touched.

## 3. It keeps the repo explainable

Every line here is one I can account for in a conversation. A dependency I can only
describe at the level of its docstring would be a liability in exactly the situation this
repo is meant to help with.

## What is NOT hand-written

To be clear about where the line sits:

- **numpy and pandas** do the array and frame work. Rewriting those would be silly and
  would prove nothing.
- **`statistics.NormalDist().inv_cdf()`** from the standard library gives the service-level
  z-score. The inverse normal CDF is a numerical approximation with a page of constants;
  hand-rolling it would add magic numbers and no insight.

The rule I used: write out the thing the repo is about, and use libraries for everything
around it.
