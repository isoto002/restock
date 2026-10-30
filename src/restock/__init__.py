"""restock — demand forecasting and reorder-point planning for inventory teams."""

from __future__ import annotations

from restock.forecast import (
    HoltWintersFit,
    InsufficientHistoryError,
    holt_winters,
    moving_average,
)
from restock.metrics import Accuracy, accuracy
from restock.policy import ReorderPolicy, days_of_cover, reorder_policy, service_level_z
from restock.simulate import SkuProfile, generate_sales

__version__ = "0.1.0"

__all__ = [
    "Accuracy",
    "HoltWintersFit",
    "InsufficientHistoryError",
    "ReorderPolicy",
    "SkuProfile",
    "__version__",
    "accuracy",
    "days_of_cover",
    "generate_sales",
    "holt_winters",
    "moving_average",
    "reorder_policy",
    "service_level_z",
]
