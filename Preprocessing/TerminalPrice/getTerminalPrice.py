import math

from impactUtils.LastPriceBuckets import LastPriceBuckets


def getTerminalPrice(trades, startTS, endTS, numBuckets):
    if numBuckets <= 0:
        raise ValueError("numBuckets must be positive.")

    terminal_buckets = LastPriceBuckets(trades, numBuckets, startTS, endTS)
    for i in range(terminal_buckets.getN() - 1, -1, -1):
        price = terminal_buckets.getPrice(i)
        if price is None:
            continue
        price = float(price)
        if math.isfinite(price):
            return price
    return float("nan")
