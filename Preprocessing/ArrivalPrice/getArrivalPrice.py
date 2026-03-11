import math

from impactUtils.FirstPriceBuckets import FirstPriceBuckets


def getArrivalPrice(trades, startTS, endTS, numBuckets):
    if numBuckets <= 0:
        raise ValueError("numBuckets must be positive.")

    firstPriceBuckets = FirstPriceBuckets(trades, numBuckets, startTS, endTS)
    for i in range(firstPriceBuckets.getN()):
        price = firstPriceBuckets.getPrice(i)
        if price is None:
            continue
        price = float(price)
        if math.isfinite(price):
            return price
    return float("nan")
