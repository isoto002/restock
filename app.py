"""Streamlit dashboard for restock.

Run it with:

    streamlit run app.py

Everything here is presentation. The forecasting and the reorder maths live in
`src/restock/` and are covered by the test suite; this file only arranges them on a page.
That separation is deliberate — a dashboard is a bad place to hide business logic, because
nothing can test it there.
"""

from __future__ import annotations

import io
import re
import warnings
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


#: The three roles a sales file has to fill, whatever its own columns are called.
ROLES = ("date", "sku", "units")

#: Header names that mean each role, in the exports people actually arrive with: Shopify,
#: Amazon Seller Central, Square, QuickBooks, and whatever a BI tool exported last. Matched
#: on a header stripped of punctuation, so "Product variant SKU" tries "product variant sku".
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "day", "order date", "purchase date", "invoice date", "period", "week"),
    "sku": (
        "sku",
        "item",
        "product",
        "asin",
        "variant",
        "product variant sku",
        "item id",
        "product id",
        "style",
        "part number",
        "material",
    ),
    "units": (
        "units",
        "qty",
        "quantity",
        "sales",
        "units ordered",
        "net items sold",
        "units sold",
        "quantity sold",
        "volume",
        "count",
        "demand",
    ),
}

#: Holt-Winters needs two full seasonal cycles, and the backtest holds out another window.
MIN_ROWS_PER_SKU = WEEKLY_CYCLE * 2 + BACKTEST_DAYS


class UploadError(ValueError):
    """An uploaded file could not be used, with a reason worth showing a person."""


@st.cache_data
def load_history(days: int, seed: int) -> pd.DataFrame:
    """Sales history, cached so changing an unrelated control does not regenerate it."""
    return generate_sales(days=days, seed=seed)


def tidy(header: object) -> str:
    """A header reduced to lowercase words, so punctuation and spacing stop mattering."""
    return re.sub(r"[^a-z0-9]+", " ", str(header).strip().lower()).strip()


def read_csv(raw: bytes) -> pd.DataFrame:
    """The uploaded bytes as a frame, with nothing assumed about its columns yet."""
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        # Deliberately broad: pandas raises a dozen different things for a malformed file,
        # and every one of them is the same story to the person who uploaded it.
        raise UploadError(f"That file could not be read as a CSV. {exc}") from exc
    if frame.empty or not len(frame.columns):
        raise UploadError("That file has no rows in it.")
    return frame


def _score(column: str, values: pd.Series, role: str) -> int:
    """How well one column fits one role. Higher wins; 0 means it does not fit at all.

    Name first, because a header that says `units` is better evidence than any amount of
    type sniffing. Content second, so a file with headers nobody has seen before still
    works: dates parse as dates, quantities parse as numbers, and the product code is
    whatever is left that repeats.
    """
    name = tidy(column)
    if name in COLUMN_ALIASES[role]:
        return 100
    if any(alias in name for alias in COLUMN_ALIASES[role]):
        return 80

    sample = values.dropna().head(200)
    if sample.empty:
        return 0
    if role == "date":
        with warnings.catch_warnings():
            # This is a probe, not a parse. Pandas warning that it had to guess the format
            # is exactly the answer being asked for, and is noise in anyone's test output.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return 60 if parsed.notna().mean() > 0.9 else 0
    if role == "units":
        numeric = pd.to_numeric(sample, errors="coerce")
        return 50 if numeric.notna().mean() > 0.9 else 0
    # A product code repeats and is not a number. Fewer distinct values scores higher.
    if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9:
        return 0
    return 40 if sample.nunique() < len(sample) else 20


def guess_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    """Best guess at which column fills each role. None when nothing plausibly fits.

    Assigned highest-confidence first and each column used once, so a single column cannot
    be claimed by two roles.
    """
    scores = [(_score(c, frame[c], role), role, c) for role in ROLES for c in frame.columns]
    chosen: dict[str, str | None] = dict.fromkeys(ROLES)
    taken: set[str] = set()
    for score, role, column in sorted(scores, key=lambda s: -s[0]):
        if score > 0 and chosen[role] is None and column not in taken:
            chosen[role], _ = column, taken.add(column)
    return chosen


def read_uploaded_history(raw: bytes, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Parse an uploaded sales CSV into the frame the rest of the page expects.

    Args:
        raw: The uploaded file's bytes.
        mapping: Which column fills each role. Guessed from the file when not given.

    Every failure raises `UploadError` with a sentence naming what is wrong and what to do.
    Somebody arriving with their own export has no idea what this app wants, so a bare
    traceback would end the visit.
    """
    source = read_csv(raw)
    chosen = mapping or guess_columns(source)

    unfilled = [role for role in ROLES if not chosen.get(role)]
    if unfilled:
        raise UploadError(
            f"Could not work out which column holds the {', '.join(unfilled)}. "
            f"Found: {', '.join(str(c) for c in source.columns)}. "
            "Pick the right ones in the sidebar."
        )

    frame = source[[chosen["date"], chosen["sku"], chosen["units"]]].copy()
    frame.columns = list(ROLES)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["units"] = pd.to_numeric(frame["units"], errors="coerce")
    frame = frame.dropna(subset=["date", "units"])
    if frame.empty:
        raise UploadError("No usable rows. Check that `date` holds dates and `units` numbers.")

    frame["sku"] = frame["sku"].astype(str)
    # One row per SKU per day. A transaction-level export has many rows per day, and the
    # forecaster expects a daily series, so sum them rather than silently taking the last.
    frame = frame.groupby(["sku", "date"], as_index=False)["units"].sum()
    frame = frame.sort_values(["sku", "date"]).reset_index(drop=True)

    long_enough = frame.groupby("sku")["units"].transform("size") >= MIN_ROWS_PER_SKU
    usable = frame[long_enough]
    if usable.empty:
        raise UploadError(
            f"Every product has fewer than {MIN_ROWS_PER_SKU} days of history. "
            "Seasonal forecasting needs two full weekly cycles plus a window to test on."
        )
    return usable.reset_index(drop=True)


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
        # Decision first. On a narrow screen the table scrolls sideways, so the answer
        # has to be in the columns you see without scrolling, not the last one.
        rows.append(
            {
                "SKU": sku,
                "Order now": "YES" if on_hand <= policy.reorder_point else "no",
                "Days of cover": round(days_of_cover(on_hand, demand_mean), 1),
                "On hand": round(on_hand),
                "Reorder point": round(policy.reorder_point),
                "Avg/day": round(demand_mean, 1),
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


#: What each role means, in the words of someone who has never read this code.
ROLE_HELP = {
    "date": "The day the sale happened.",
    "sku": "The product code or name.",
    "units": "How many were sold.",
}


def column_mapping_controls(source: pd.DataFrame) -> dict[str, str]:
    """Show which column was matched to each role, and let it be corrected.

    The guess is usually right, so this stays collapsed. It exists because a wrong guess
    that cannot be fixed is worse than no guess at all, and because seeing the match is
    what makes somebody trust the numbers underneath.
    """
    guess = guess_columns(source)
    options = [str(c) for c in source.columns]
    mapping: dict[str, str] = {}

    with st.sidebar.expander("Columns", expanded=any(v is None for v in guess.values())):
        st.caption("Matched from your file. Change any that are wrong.")
        for role in ROLES:
            current = guess.get(role)
            mapping[role] = st.selectbox(
                role,
                options,
                index=options.index(str(current)) if current in options else 0,
                help=ROLE_HELP[role],
                key=f"column-{role}",
            )
    return mapping


def pick_history() -> tuple[pd.DataFrame, str]:
    """The uploaded file when there is one, the generated sample otherwise.

    Returns `(history, caption)`. A failed upload falls back to the sample rather than
    emptying the page, because a visitor whose CSV was rejected should still see the thing
    working while they fix their file.
    """
    st.sidebar.header("Your data")
    upload = st.sidebar.file_uploader(
        "Sales history (CSV)",
        type="csv",
        help="Three columns: date, sku, units. One row per product per day.",
    )
    st.sidebar.caption("Nothing is stored. The file is read in memory and discarded.")

    if upload is None:
        return (
            load_history(days=DEMO_DAYS, seed=DEMO_SEED),
            "Demand forecasting and reorder-point planning. Showing generated sample data "
            "— upload your own sales CSV in the sidebar to plan against it.",
        )

    try:
        source = read_csv(upload.getvalue())
        mapping = column_mapping_controls(source)
        history = read_uploaded_history(upload.getvalue(), mapping)
    except UploadError as exc:
        st.sidebar.error(str(exc))
        return (
            load_history(days=DEMO_DAYS, seed=DEMO_SEED),
            "Showing sample data, because the uploaded file could not be used. "
            "The reason is in the sidebar.",
        )

    skus = history["sku"].nunique()
    days = history["date"].nunique()
    return history, f"Planning against your upload: {skus} products across {days} days."


def main() -> None:
    """Draw the page."""
    st.title("📦 restock")

    lead_time, lead_time_sd, service_level = sidebar_controls()
    history, caption = pick_history()
    st.caption(caption)

    draw_at_risk(history, lead_time, lead_time_sd, service_level)
    draw_forecast(history)


if __name__ == "__main__":
    main()
