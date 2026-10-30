"""Streamlit dashboard for restock.

Run it with:

    streamlit run app.py

Everything here is presentation. The forecasting and the reorder maths live in
`src/restock/` and are covered by the test suite; this file only arranges them on a page.
That separation is deliberate — a dashboard is a bad place to hide business logic, because
nothing can test it there.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import streamlit as st

from restock.forecast import holt_winters
from restock.metrics import accuracy
from restock.policy import days_of_cover, reorder_policy
from restock.simulate import generate_sales

WEEKLY_CYCLE = 7
BACKTEST_DAYS = 28

#: Two years of demo history — Holt-Winters needs at least two full seasonal cycles.
DEMO_DAYS = 730
DEMO_SEED = 7

#: How much history the chart shows. Long enough to see the weekly shape, short enough
#: that the forecast is not a sliver at the right-hand edge.
CHART_HISTORY_DAYS = 90
FORECAST_HORIZON = 28

#: Stand-in stock levels are spread this far either side of each reorder point, so the
#: at-risk table shows a mix of "order now" and "fine" rather than one uniform answer.
STOCK_POSITION_RANGE = (0.6, 1.8)

st.set_page_config(page_title="restock", page_icon="📦", layout="centered")


@st.cache_data
def load_history(days: int, seed: int) -> pd.DataFrame:
    """Sales history, cached so changing an unrelated control does not regenerate it."""
    return generate_sales(days=days, seed=seed)


@st.cache_data
def forecast_sku(units: tuple[float, ...], horizon: int) -> tuple[list[float], float, float]:
    """Forecast one SKU, cached on the series itself.

    Streamlit re-runs this whole script on every interaction, so without caching, moving
    any control would refit every model from scratch. The input is a tuple because the
    cache key has to be hashable.

    Returns `(forecast, backtest_mape, bias)`.
    """
    series = np.asarray(units, dtype=float)
    train, test = series[:-BACKTEST_DAYS], series[-BACKTEST_DAYS:]

    scored = accuracy(
        actual=test,
        forecast=holt_winters(train, season_length=WEEKLY_CYCLE).predict(BACKTEST_DAYS),
    )
    forecast = holt_winters(series, season_length=WEEKLY_CYCLE).predict(horizon)
    return list(forecast), scored.mape, scored.bias


def sidebar_controls() -> tuple[float, float, float]:
    """The planning assumptions. Every control carries a plain-English explanation."""
    st.sidebar.header("Planning assumptions")

    lead_time = st.sidebar.slider(
        "Supplier lead time (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="How long between placing a purchase order and the stock arriving.",
    )
    lead_time_sd = st.sidebar.slider(
        "Lead time variability (days)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help=(
            "How much that lead time swings. This is the figure most teams never measure, "
            "and it usually drives more buffer than demand variability does."
        ),
    )
    service_level = st.sidebar.slider(
        "Service level",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help=(
            "The share of replenishment cycles you want to get through without a stockout. "
            "Higher means more buffer and more cash tied up in stock."
        ),
    )
    return float(lead_time), float(lead_time_sd), float(service_level)


def at_risk_table(
    history: pd.DataFrame, lead_time: float, lead_time_sd: float, service_level: float
) -> pd.DataFrame:
    """Every SKU, sorted by how soon it runs out."""
    rows = []
    for sku, group in history.groupby("sku", sort=True):
        series = group["units"].to_numpy(dtype=float)
        demand_mean = float(np.mean(series))

        policy = reorder_policy(
            demand_mean=demand_mean,
            demand_sd=float(np.std(series, ddof=1)),
            lead_time_days=lead_time,
            service_level=service_level,
            lead_time_sd=lead_time_sd,
        )
        # Stand-in for a live stock figure, which a real deployment would read from an ERP.
        # Seeded on the SKU name so the demo is stable between reruns, and spread either
        # side of the reorder point so the table actually shows the decision it exists to
        # make. Every SKU comfortably in stock would be a reassuring but useless demo.
        # crc32, not hash(): Python salts string hashes per process, so hash() would move
        # the whole table every time the server restarted.
        seed = zlib.crc32(str(sku).encode("utf-8"))
        position = np.random.default_rng(seed).uniform(*STOCK_POSITION_RANGE)
        on_hand = policy.reorder_point * float(position)
        rows.append(
            {
                "SKU": sku,
                "On hand": round(on_hand),
                "Avg/day": round(demand_mean, 1),
                "Days of cover": round(days_of_cover(on_hand, demand_mean), 1),
                "Reorder point": round(policy.reorder_point),
                "Order now": "YES" if on_hand <= policy.reorder_point else "no",
            }
        )
    return pd.DataFrame(rows).sort_values("Days of cover").reset_index(drop=True)


def draw_at_risk(
    history: pd.DataFrame, lead_time: float, lead_time_sd: float, service_level: float
) -> None:
    """The headline section: which products need ordering, soonest first."""
    st.subheader("What needs ordering")
    st.dataframe(
        at_risk_table(history, lead_time, lead_time_sd, service_level),
        width="stretch",
        hide_index=True,
    )


def history_and_forecast_chart(group: pd.DataFrame, forecast: list[float]) -> pd.DataFrame:
    """Recent actuals and the forecast on one date index, ready for `st.line_chart`."""
    recent = group.tail(CHART_HISTORY_DAYS)
    future_dates = pd.date_range(recent["date"].iloc[-1], periods=len(forecast) + 1, freq="D")[1:]
    return pd.concat(
        [
            pd.DataFrame({"date": recent["date"], "Actual": recent["units"].to_numpy()}),
            pd.DataFrame({"date": future_dates, "Forecast": forecast}),
        ]
    ).set_index("date")


def draw_forecast(history: pd.DataFrame) -> None:
    """One product's demand history, its forecast, and how accurate the model has been."""
    st.subheader("Forecast")
    sku = st.selectbox(
        "Product",
        sorted(history["sku"].unique()),
        help="Pick a product to see its recent demand and the forecast for the weeks ahead.",
    )

    group = history[history["sku"] == sku]
    units = tuple(group["units"].to_numpy(dtype=float))
    forecast, mape, bias = forecast_sku(units, horizon=FORECAST_HORIZON)

    left, right = st.columns(2)
    left.metric(
        "Backtest error (MAPE)",
        f"{mape:.1f}%",
        help="Measured on weeks the model never saw.",
    )
    right.metric(
        "Bias (units/day)",
        f"{bias:+.1f}",
        help="Positive means the forecast runs high.",
    )

    st.line_chart(history_and_forecast_chart(group, forecast))
    st.caption(
        "A daily forecast for noisy retail demand is routinely 20-30% off. That is not a "
        "broken model — it is why safety stock exists."
    )


def main() -> None:
    """Draw the page."""
    st.title("📦 restock")
    st.caption("Demand forecasting and reorder-point planning. Runs on generated sample data.")

    lead_time, lead_time_sd, service_level = sidebar_controls()
    history = load_history(days=DEMO_DAYS, seed=DEMO_SEED)

    draw_at_risk(history, lead_time, lead_time_sd, service_level)
    draw_forecast(history)


if __name__ == "__main__":
    main()
