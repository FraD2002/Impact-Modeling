import math


def getDailyValue(trades):
    total_daily_value = 0.0
    for i in range(0, trades.getN()):
        size = float(trades.getSize(i))
        price = float(trades.getPrice(i))
        if not math.isfinite(size) or not math.isfinite(price):
            continue
        if size < 0:
            continue
        total_daily_value += size * price

    return total_daily_value
