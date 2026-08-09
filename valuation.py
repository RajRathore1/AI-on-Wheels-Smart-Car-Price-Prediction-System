"""Analytics that turn a single predicted number into something a seller can act on:
how confident the estimate is, where it sits in the market, how it ages, and what
comparable cars actually sold for."""

import json
import os

import numpy as np
import pandas as pd

from condition_assessment import DAMAGE_SEVERITY, POINTS_PER_DETECTION

CALIBRATION_PATH = os.path.join("models", "interval_calibration.json")

# Fallback if the calibration file is missing; measured on held-out data.
_DEFAULT_CALIBRATION = {"band_80": 0.40, "medape": 0.187, "within_20pct": 0.52}


def _load_calibration():
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return _DEFAULT_CALIBRATION


_CALIBRATION = _load_calibration()

# How much the trees typically disagree, used to rank one car against another.
_TYPICAL_SPREAD = 0.55


def predict_with_range(pipe, input_df: pd.DataFrame):
    """Point estimate plus a range that means what it claims.

    The forest's internal disagreement is tempting to use as the interval, but measured
    against held-out cars it only contains the true price 58% of the time while looking
    like an 80% range. So the width comes from the model's actual error distribution
    (calibrated so 80% of real cars fall inside), and the tree spread is used only for
    what it's genuinely good at: telling which cars are harder than others.
    """
    transformed = pipe[:-1].transform(input_df)
    forest = pipe[-1]

    tree_preds = np.array([est.predict(transformed)[0] for est in forest.estimators_])
    point = float(tree_preds.mean())

    band = _CALIBRATION.get("band_80", 0.40)
    low = point * (1 - band)
    high = point * (1 + band)

    # Relative disagreement between trees -- weak but real signal (r≈0.27 with true error)
    p10, p90 = np.percentile(tree_preds, [10, 90])
    spread = float((p90 - p10) / point) if point else 1.0

    if spread < _TYPICAL_SPREAD * 0.6:
        confidence = "High"
    elif spread < _TYPICAL_SPREAD * 1.2:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "price": point,
        "low": low,
        "high": high,
        "confidence": confidence,
        "spread": spread,
        "band": band,
        "coverage": 0.80,
    }


def depreciation_curve(pipe, base_row: dict, span: int = 8) -> pd.DataFrame:
    """What the same car is worth across manufacturing years, per the model.

    This reads value off the model's learned age curve rather than extrapolating into
    the future, so it stays inside what the training data actually supports.
    """
    years = list(range(max(1995, base_row["year"] - span), min(2025, base_row["year"] + 3) + 1))
    rows = []
    for y in years:
        r = dict(base_row)
        r["year"] = y
        rows.append(r)
    preds = pipe.predict(pd.DataFrame(rows))
    return pd.DataFrame({"year": years, "price": preds})


def similar_listings(df: pd.DataFrame, company: str, name: str, year: int,
                     kms: int, limit: int = 6) -> pd.DataFrame:
    """Real listings closest to this car, nearest first.

    Prefers the exact model and widens to the brand only if there aren't enough, so
    the comparison stays meaningful instead of padding with unrelated cars.
    """
    pool = df[(df["company"] == company) & (df["name"] == name)]
    if len(pool) < 3:
        pool = df[df["company"] == company]
    if pool.empty:
        return pool

    pool = pool.copy()
    # Normalised distance so kilometres (tens of thousands) don't swamp years.
    year_gap = (pool["year"] - year).abs() / 10.0
    kms_gap = (pool["kms_driven"] - kms).abs() / 50000.0
    pool["_distance"] = year_gap + kms_gap

    out = pool.nsmallest(limit, "_distance")
    return out[["name", "year", "kms_driven", "fuel_type", "Price"]].reset_index(drop=True)


def market_position(df: pd.DataFrame, company: str, price: float):
    """Where this price sits among comparable listings, as a percentile."""
    pool = df[df["company"] == company]["Price"]
    if len(pool) < 5:
        pool = df["Price"]
    percentile = float((pool < price).mean() * 100)
    return {"percentile": percentile, "pool": pool, "median": float(pool.median())}


def damage_impact(condition_result, base_price: float):
    """Rupee cost attributed to each damage type.

    Splits the condition deduction across findings in the same proportion the score
    used, so the breakdown always reconciles with the headline adjustment.
    """
    if not condition_result or not condition_result.get("summary"):
        return []

    total_lost = base_price * (1 - condition_result["price_multiplier"])
    weights = []
    for item in condition_result["summary"]:
        w = DAMAGE_SEVERITY.get(item["class"], 0.5) * POINTS_PER_DETECTION * item["count"]
        weights.append(w)

    weight_sum = sum(weights) or 1.0
    return [
        {"class": item["class"], "count": item["count"], "cost": total_lost * w / weight_sum}
        for item, w in zip(condition_result["summary"], weights)
    ]


def build_report(car: dict, valuation: dict, condition_result, impacts) -> str:
    """Plain-text summary the user can save or send to a buyer."""
    lines = [
        "CAR VALUATION REPORT",
        "=" * 46,
        "",
        f"Vehicle      : {car['name']}",
        f"Brand        : {car['company']}",
        f"Year         : {car['year']}",
        f"Kilometers   : {car['kms_driven']:,}",
        f"Fuel type    : {car['fuel_type']}",
        "",
        "-" * 46,
        f"Estimated value : Rs {valuation['price']:,.0f}",
        f"Likely range    : Rs {valuation['low']:,.0f} - Rs {valuation['high']:,.0f}",
        f"Confidence      : {valuation['confidence']}",
    ]

    if condition_result:
        score = condition_result["condition_score"]
        lines += [
            "",
            "-" * 46,
            "CONDITION (from photos)",
            f"Score : {score}/100",
        ]
        if impacts:
            lines.append("")
            lines.append("Damage found:")
            for imp in impacts:
                times = f" x{imp['count']}" if imp["count"] > 1 else ""
                lines.append(f"  - {imp['class']}{times}: approx -Rs {imp['cost']:,.0f}")
        else:
            lines.append("No visible damage detected.")

    lines += [
        "",
        "-" * 46,
        "Estimates are guidance only. Condition, service history and",
        "location all affect the final selling price.",
    ]
    return "\n".join(lines)
