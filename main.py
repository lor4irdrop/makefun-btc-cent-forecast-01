#!/usr/bin/env python3
"""BTC 4h forecaster v2: log-linear trend + 1h momentum + 6h mean reversion.

Reads one case file per run (4,320 one-minute closes, last = checkpoint P)
and writes one forecast for the price 240 minutes after the checkpoint.
Standard library only, deterministic, no network.
"""
import json
import math
import os
from pathlib import Path

TREND_WINDOW = 720
TREND_DAMP = 0.45
MOMENTUM_WEIGHT = 0.4
REVERSION_WEIGHT = -0.2
MEAN_WINDOW = 360
MEAN_PULL = 0.05
CLAMP_PCT = 0.015
HORIZON_MINUTES = 240


def ols_slope(values):
    """Slope per minute of an OLS line fit to log prices."""
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - mean_x
        num += dx * (y - mean_y)
        den += dx * dx
    return num / den if den else 0.0


def forecast_price(prices):
    """One forecast from the full close series; P is the last close."""
    p = prices[-1]
    n = len(prices)
    trend_n = min(TREND_WINDOW, n)
    mean_n = min(MEAN_WINDOW, n)
    back60 = min(60, n - 1)
    back360 = min(360, n - 1)
    logs = [math.log(x) for x in prices]
    slope = ols_slope(logs[-trend_n:])
    r60 = logs[-1] - logs[-1 - back60]
    r360 = logs[-1] - logs[-1 - back360]
    mean360 = math.fsum(prices[-mean_n:]) / mean_n
    drift = TREND_DAMP * HORIZON_MINUTES * slope
    drift += MOMENTUM_WEIGHT * r60 + REVERSION_WEIGHT * r360
    guess = p * math.exp(drift) + MEAN_PULL * (mean360 - p)
    low = p * (1.0 - CLAMP_PCT)
    high = p * (1.0 + CLAMP_PCT)
    if guess < low:
        guess = low
    elif guess > high:
        guess = high
    if not math.isfinite(guess) or guess <= 0.0 or guess >= 10000000.0:
        guess = p
    return round(guess, 5)


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for in_file in sorted(in_dir.glob("*.in")):
        case = json.loads(in_file.read_text(encoding="utf-8"))
        prices = case["markets"][0]["prices"]
        price = forecast_price(prices)
        payload = {"forecasts": [{"case": case["case"], "price": price}]}
        out_file = out_dir / (in_file.stem + ".out")
        out_file.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
