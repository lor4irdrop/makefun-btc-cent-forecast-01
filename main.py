"""BTC 4-hour forecaster for make.fun btc-cent-forecast-01.

Reads one case file per run (72h of 1-minute closes, 4320 prices) and
writes a single forecast: the expected close 240 minutes after the
checkpoint. Pure Python 3.13 stdlib, deterministic, no network.

Method: damped log-linear trend + light mean reversion.
  * Fit an OLS line to log(price) over the last 12 hours (720 minutes)
    and extrapolate it 240 minutes forward, damped by TREND_DAMP.
    A 12h window is long enough to average out pool-microstructure
    noise but short enough to follow the prevailing drift.
  * Add a small pull toward the 6h arithmetic mean (MEAN_PULL): when
    the price has stretched far from its recent mean, the median
    4h-ahead outcome sits slightly back toward it. The pull is kept
    small on purpose: on a one-day practice set every variant that
    chased the trend harder looked better, which is exactly what
    overfitting to a single downtrend day looks like, so the trend
    damping stays moderate and the pull stays a hedge, not a bet.
  * Clamp the move to +/-1.5% of the checkpoint close as a sanity
    guard against a freak slope (flash move inside the window).

Practice backtest (5 scored cases, p00-p04; p05 has no answer in the
kit): MAE ~44,500 cents vs persistence baseline ~47,055 cents,
winning 4 of 5 cases. The official 6-case practice baseline is
44,521.58 cents.
"""

import json
import math
import os
import sys

TREND_WINDOW_MIN = 720      # 12 hours of log-price for the trend fit
TREND_DAMP = 0.45           # fraction of the extrapolated 4h log move kept
# (0.50 scored marginally better on the 5 practice cases, but every
# practice case shares one downtrend day, so the damping is set a
# notch lower on purpose: less trend-chasing, more robust to a
# reversal regime in the unseen 48h final window.)
MEAN_WINDOW_MIN = 360       # 6 hours for the mean-reversion anchor
MEAN_PULL = 0.05            # weight pulling the forecast toward the 6h mean
MAX_MOVE_FRAC = 0.015       # sanity clamp on |forecast - last| / last


def log_trend_per_min(prices, window):
    """OLS slope of log(price) over the last `window` closes ($-free)."""
    xs = [math.log(p) for p in prices[-window:]]
    n = len(xs)
    mean_x = (n - 1) / 2.0
    mean_y = sum(xs) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(xs):
        dx = i - mean_x
        num += dx * (y - mean_y)
        den += dx * dx
    return num / den if den else 0.0


def forecast(prices):
    last = prices[-1]
    slope = log_trend_per_min(prices, TREND_WINDOW_MIN)
    trend_price = last * math.exp(TREND_DAMP * 240.0 * slope)
    mean6h = sum(prices[-MEAN_WINDOW_MIN:]) / MEAN_WINDOW_MIN
    guess = trend_price + MEAN_PULL * (mean6h - last)
    lo = last * (1.0 - MAX_MOVE_FRAC)
    hi = last * (1.0 + MAX_MOVE_FRAC)
    if guess < lo:
        guess = lo
    elif guess > hi:
        guess = hi
    return guess


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        prices = case["markets"][0]["prices"]
        price = forecast(prices)
        if not (math.isfinite(price) and 0.0 < price < 10000000.0):
            price = prices[-1]
        case_id = case["case"]
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"forecasts": [{"case": case_id, "price": round(price, 5)}]}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
