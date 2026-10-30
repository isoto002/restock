"""Command line for restock.

Two commands:

    restock demo                       generate a history and plan against it
    restock plan --input sales.csv     plan against your own history

The accuracy figure is a genuine backtest, not an in-sample score: the model is fitted on
all but the last few weeks and then measured against weeks it never saw. An in-sample
number flatters every forecaster and is close to meaningless.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from restock import __version__
from restock.forecast import InsufficientHistoryError, holt_winters, moving_average
from restock.metrics import Accuracy, accuracy
from restock.policy import ReorderPolicy, reorder_policy, service_level_z
from restock.simulate import generate_sales

REQUIRED_COLUMNS = ("date", "sku", "units")
WEEKLY_CYCLE = 7
DEFAULT_HOLDOUT_DAYS = 28
EXIT_OK = 0
EXIT_USER_ERROR = 2


class UserError(Exception):
    """Something the caller can fix — reported as a plain message, never a traceback."""


@dataclass(frozen=True)
class SkuPlan:
    """One row of the reorder plan."""

    sku: str
    demand_mean: float
    forecast_next_week: float
    policy: ReorderPolicy
    backtest: Accuracy
    baseline: Accuracy


def _load_sales(path: Path) -> pd.DataFrame:
    """Read a sales CSV, or raise `UserError` explaining what is wrong with it."""
    if not path.exists():
        raise UserError(f"Input file not found: {path}")

    try:
        frame = pd.read_csv(path, parse_dates=["date"])
    except ValueError as exc:
        raise UserError(
            f"Could not read {path}. Expected columns {', '.join(REQUIRED_COLUMNS)}. ({exc})"
        ) from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise UserError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )
    return frame.sort_values(["sku", "date"])


def _backtest(series: np.ndarray, holdout: int) -> tuple[Accuracy, Accuracy]:
    """Score Holt-Winters and the naive baseline on periods neither model ever saw.

    Returns `(model, baseline)`. The baseline is a flat seven-day average carried
    forward. It is here because a forecaster that cannot beat it is not earning its
    keep, and that comparison is the first thing a reviewer should be able to check.
    """
    train, test = series[:-holdout], series[-holdout:]

    fit = holt_winters(train, season_length=WEEKLY_CYCLE)
    model = accuracy(actual=test, forecast=fit.predict(holdout))

    flat = np.full(holdout, moving_average(train, window=WEEKLY_CYCLE))
    baseline = accuracy(actual=test, forecast=flat)
    return model, baseline


def _plan_for_sku(
    sku: str, series: np.ndarray, *, lead_time: float, service_level: float, lead_time_sd: float
) -> SkuPlan:
    """Build the full plan for one SKU, or raise `UserError` if its history is too short."""
    holdout = min(DEFAULT_HOLDOUT_DAYS, max(WEEKLY_CYCLE, len(series) // 4))
    try:
        backtest, baseline = _backtest(series, holdout)
        fit = holt_winters(series, season_length=WEEKLY_CYCLE)
    except InsufficientHistoryError as exc:
        raise UserError(
            f"{sku}: not enough history to forecast. {exc} "
            f"At least {2 * WEEKLY_CYCLE + holdout} days per SKU is a safe minimum."
        ) from exc

    next_week = fit.predict(WEEKLY_CYCLE)
    policy = reorder_policy(
        demand_mean=float(np.mean(series)),
        demand_sd=float(np.std(series, ddof=1)),
        lead_time_days=lead_time,
        service_level=service_level,
        lead_time_sd=lead_time_sd,
    )
    return SkuPlan(
        sku, float(np.mean(series)), float(np.sum(next_week)), policy, backtest, baseline
    )


def _build_plans(frame: pd.DataFrame, **kwargs: float) -> list[SkuPlan]:
    """One `SkuPlan` per SKU in the frame, in alphabetical order."""
    return [
        _plan_for_sku(str(sku), group["units"].to_numpy(dtype=float), **kwargs)  # type: ignore[arg-type]
        for sku, group in frame.groupby("sku", sort=True)
    ]


def _render(plans: Sequence[SkuPlan], *, lead_time: float, service_level: float) -> str:
    """Format the plan as a plain-text table."""
    header = (
        f"{'SKU':<16}{'Avg/day':>10}{'Next 7d':>10}"
        f"{'Safety':>10}{'Reorder pt':>13}{'MAPE':>9}{'Naive':>9}{'Bias':>9}"
    )
    lines = [
        "",
        f"REORDER PLAN  -  lead time {lead_time:g} days, service level {service_level:.0%}",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for plan in plans:
        lines.append(
            f"{plan.sku:<16}{plan.demand_mean:>10.1f}{plan.forecast_next_week:>10.0f}"
            f"{plan.policy.safety_stock:>10.0f}{plan.policy.reorder_point:>13.0f}"
            f"{plan.backtest.mape:>8.1f}%{plan.baseline.mape:>8.1f}%{plan.backtest.bias:>9.1f}"
        )
    lines.extend(
        [
            "",
            "Reorder point = expected demand over the lead time + safety stock.",
            "MAPE and Bias come from a backtest on weeks the model never saw.",
            "Naive is a flat 7-day average over those same weeks - the bar to beat.",
            "",
        ]
    )
    return "\n".join(lines)


def _run(frame: pd.DataFrame, args: argparse.Namespace) -> int:
    """Shared tail of both commands: plan, render, print."""
    plans = _build_plans(
        frame,
        lead_time=args.lead_time,
        service_level=args.service_level,
        lead_time_sd=args.lead_time_sd,
    )
    print(_render(plans, lead_time=args.lead_time, service_level=args.service_level))
    return EXIT_OK


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Options both commands take."""
    parser.add_argument(
        "--lead-time", type=float, default=7.0, help="Average supplier lead time in days."
    )
    parser.add_argument(
        "--lead-time-sd",
        type=float,
        default=0.0,
        help="Standard deviation of the lead time in days. Measure it before assuming zero.",
    )
    parser.add_argument(
        "--service-level",
        type=float,
        default=0.95,
        help="Share of cycles to get through without a stockout, between 0 and 1.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="restock", description="Demand forecasting and reorder-point planning."
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command")

    demo = subcommands.add_parser("demo", help="Generate a sample history and plan against it.")
    _add_shared_arguments(demo)

    plan = subcommands.add_parser("plan", help="Plan against your own sales CSV.")
    plan.add_argument(
        "--input", required=True, type=Path, help="CSV with columns: date, sku, units."
    )
    _add_shared_arguments(plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for user error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage(sys.stderr)
        print("error: pick a command — 'demo' or 'plan'.", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        service_level_z(args.service_level)
        frame = generate_sales() if args.command == "demo" else _load_sales(args.input)
        return _run(frame, args)
    except (UserError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
